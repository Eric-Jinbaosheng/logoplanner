from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from checkpoint_utils import load_checkpoint_payload, load_weights_into_model
from dataset_interndata_n1 import InternDataN1TarDataset
from policy_network import LoGoPlanner_Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoGoPlanner directly from official InternData-N1 traj_data tar files.")
    parser.add_argument("--data", nargs="+", required=True, help="One or more InternData-N1 .tar.gz files or directories.")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output path.")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional pretrained checkpoint to initialize model weights without restoring optimizer state.",
    )
    parser.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume from.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=12)
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-episodes-per-tar", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--target-camera-height", type=float, default=None)
    parser.add_argument("--target-camera-pitch-deg", type=float, default=None)
    parser.add_argument("--camera-height-tol", type=float, default=0.05)
    parser.add_argument("--camera-pitch-tol-deg", type=float, default=3.0)
    parser.add_argument(
        "--episode-cache-dir",
        type=str,
        default="",
        help="Optional directory for cached per-episode decoded InternData-N1 npz files.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0, help="If > 0, stop training after this many optimizer steps.")
    parser.add_argument("--lambda-noise", type=float, default=1.0)
    parser.add_argument("--lambda-camera", type=float, default=0.0)
    parser.add_argument("--lambda-local", type=float, default=0.0)
    parser.add_argument("--lambda-world", type=float, default=0.0)
    parser.add_argument("--lambda-value", type=float, default=0.0)
    parser.add_argument("--lact-inner-steps", type=int, default=1, help="Inner-loop update steps for LaCT decoder blocks.")
    parser.add_argument("--seed", type=int, default=20260320)
    return parser.parse_args()


def collate_fn(samples: list[dict]) -> dict:
    batch = {}
    tensor_keys = [key for key, value in samples[0].items() if torch.is_tensor(value)]
    for key in tensor_keys:
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    batch["sample_path"] = [sample["sample_path"] for sample in samples]
    return batch


def move_to_device(batch: dict, device: str) -> dict:
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device) if torch.is_tensor(value) else value
    return output


def build_model(args: argparse.Namespace) -> LoGoPlanner_Policy:
    model = LoGoPlanner_Policy(
        memory_size=args.memory_size,
        context_size=args.context_size,
        predict_size=args.predict_size,
        device=args.device,
    ).to(args.device)
    return model


def set_lact_inner_steps(model: LoGoPlanner_Policy, inner_steps: int) -> None:
    for module in model.modules():
        if hasattr(module, "set_inner_steps"):
            module.set_inner_steps(inner_steps)


def encode_condition(model: LoGoPlanner_Policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    memory_rgbd = batch["memory_rgbd"]
    context_rgbd = batch["context_rgbd"]
    start_goal = batch["start_goal"]

    rgbd_embed = model.rgbd_encoder(memory_rgbd[..., :3], memory_rgbd[:, -1, ..., 3:4])
    state_outputs, geom_outputs = model.state_encoder(context_rgbd[..., :3], context_rgbd[..., 3:4])
    _, state_token, scene_token = state_outputs
    camera_poses, local_points, world_points = geom_outputs

    startgoal_embed = model.start_encoder(start_goal).unsqueeze(1)
    unify_token = torch.cat([state_token, scene_token], dim=1)
    state_embed = model.state_decoder(torch.cat([state_token, startgoal_embed], dim=1))

    aux = {
        "camera_poses": camera_poses,
        "local_points": local_points,
        "world_points": world_points,
    }
    return rgbd_embed, state_embed, unify_token, aux


def forward_loss(model: LoGoPlanner_Policy, batch: dict, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    rgbd_embed, state_embed, unify_token, aux = encode_condition(model, batch)
    target_traj = batch["target_traj"]

    timestep = torch.randint(
        low=0,
        high=model.noise_scheduler.config.num_train_timesteps,
        size=(1,),
        device=target_traj.device,
    )
    noise = torch.randn_like(target_traj)
    noisy_traj = model.noise_scheduler.add_noise(target_traj, noise, timestep.repeat(target_traj.shape[0]))
    pred_noise = model.predict_noise(noisy_traj, timestep, state_embed, rgbd_embed, unify_token)
    loss_noise = F.mse_loss(pred_noise, noise)

    total_loss = args.lambda_noise * loss_noise
    losses: dict[str, torch.Tensor] = {
        "loss": total_loss,
        "loss_noise": loss_noise,
    }

    if args.lambda_camera > 0.0 and "target_camera_pose" in batch:
        loss_camera = F.mse_loss(aux["camera_poses"], batch["target_camera_pose"])
        total_loss = total_loss + args.lambda_camera * loss_camera
        losses["loss_camera"] = loss_camera

    if args.lambda_local > 0.0 and "target_local_points" in batch:
        loss_local = F.mse_loss(aux["local_points"], batch["target_local_points"])
        total_loss = total_loss + args.lambda_local * loss_local
        losses["loss_local"] = loss_local

    if args.lambda_world > 0.0 and "target_world_points" in batch:
        loss_world = F.mse_loss(aux["world_points"], batch["target_world_points"])
        total_loss = total_loss + args.lambda_world * loss_world
        losses["loss_world"] = loss_world

    if args.lambda_value > 0.0 and "target_value" in batch:
        pred_value = model.predict_critic(target_traj, rgbd_embed, unify_token)
        loss_value = F.mse_loss(pred_value, batch["target_value"].view_as(pred_value))
        total_loss = total_loss + args.lambda_value * loss_value
        losses["loss_value"] = loss_value

    losses["loss"] = total_loss
    return losses


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = InternDataN1TarDataset(
        args.data,
        memory_size=args.memory_size,
        context_size=args.context_size,
        predict_size=args.predict_size,
        depth_scale=args.depth_scale,
        step_stride=args.step_stride,
        max_episodes_per_tar=args.max_episodes_per_tar,
        episode_cache_dir=args.episode_cache_dir or None,
        target_camera_height=args.target_camera_height,
        target_camera_pitch_deg=args.target_camera_pitch_deg,
        camera_height_tol=args.camera_height_tol,
        camera_pitch_tol_deg=args.camera_pitch_tol_deg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = build_model(args)
    set_lact_inner_steps(model, args.lact_inner_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.device.startswith("cuda"))

    start_step = 0
    if args.init_checkpoint:
        missing_keys, unexpected_keys = load_weights_into_model(
            model,
            args.init_checkpoint,
            map_location=args.device,
            strict=False,
        )
        if missing_keys or unexpected_keys:
            print(
                json.dumps(
                    {
                        "init_checkpoint": args.init_checkpoint,
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                    }
                ),
                flush=True,
            )
    if args.resume:
        missing_keys, unexpected_keys = load_weights_into_model(
            model,
            args.resume,
            map_location=args.device,
            strict=False,
        )
        state = load_checkpoint_payload(args.resume, map_location=args.device)
        if isinstance(state, dict) and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_step = int(state.get("step", 0)) if isinstance(state, dict) else 0
        if missing_keys or unexpected_keys:
            print(
                json.dumps(
                    {
                        "resume": args.resume,
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                    }
                ),
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    global_step = start_step
    stop_training = False

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break
            batch = move_to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.device.startswith("cuda")):
                loss_dict = forward_loss(model, batch, args)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if global_step % args.log_every == 0:
                metrics = {"step": global_step, "epoch": epoch}
                metrics.update({key: float(value.detach().item()) for key, value in loss_dict.items()})
                print(json.dumps(metrics), flush=True)

            if global_step > 0 and global_step % args.save_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "args": vars(args),
                    },
                    args.output.with_name(f"{args.output.stem}_step{global_step}{args.output.suffix}"),
                )

            global_step += 1
        if stop_training:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": global_step,
            "args": vars(args),
        },
        args.output,
    )
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
