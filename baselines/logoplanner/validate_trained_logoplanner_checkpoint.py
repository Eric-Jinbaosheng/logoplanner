from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from checkpoint_utils import load_weights_into_model
from dataset_interndata_n1 import InternDataN1TarDataset
from policy_network import LoGoPlanner_Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained LoGoPlanner checkpoint on one pointgoal sample.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", nargs="+", required=True, help="InternData-N1 .tar.gz files or directories.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=12)
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-episodes-per-tar", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--episode-cache-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = InternDataN1TarDataset(
        args.data,
        memory_size=args.memory_size,
        context_size=args.context_size,
        predict_size=args.predict_size,
        depth_scale=args.depth_scale,
        step_stride=args.step_stride,
        max_episodes_per_tar=args.max_episodes_per_tar,
        episode_cache_dir=args.episode_cache_dir or None,
    )
    sample = dataset[args.sample_index]

    model = LoGoPlanner_Policy(
        memory_size=args.memory_size,
        context_size=args.context_size,
        predict_size=args.predict_size,
        device=args.device,
    ).to(args.device)
    missing_keys, unexpected_keys = load_weights_into_model(
        model,
        args.checkpoint,
        map_location=args.device,
        strict=False,
    )
    model.eval()

    start_goal = sample["start_goal"].cpu().numpy()[None]
    memory_rgbd = sample["memory_rgbd"].cpu().numpy()[None]
    context_rgbd = sample["context_rgbd"].cpu().numpy()[None]
    target_traj = sample["target_traj"].cpu().numpy()

    with torch.no_grad():
        all_trajectory, critic_values, positive_trajectory, negative_trajectory, sub_pointgoal_pd = model.predict_pointgoal_action(
            start_goal=start_goal,
            memory_rgbd=memory_rgbd,
            context_rgbd=context_rgbd,
            sample_num=4,
        )

    result = {
        "sample_path": sample["sample_path"],
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "target_traj_shape": list(target_traj.shape),
        "pred_all_trajectory_shape": list(all_trajectory.shape),
        "critic_values_shape": list(np.asarray(critic_values).shape),
        "sub_pointgoal_pd_shape": list(np.asarray(sub_pointgoal_pd).shape),
        "target_traj_mean_abs": float(np.abs(target_traj).mean()),
        "pred_traj_mean_abs": float(np.abs(all_trajectory).mean()),
        "critic_mean": float(np.asarray(critic_values).mean()),
        "critic_std": float(np.asarray(critic_values).std()),
        "sub_pointgoal_pd": np.asarray(sub_pointgoal_pd).tolist(),
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
