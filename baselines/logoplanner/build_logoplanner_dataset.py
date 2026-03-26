from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


TARGET_H = 168
TARGET_W = 308


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LoGoPlanner per-step training samples from episode .npz files."
    )
    parser.add_argument("--input", nargs="+", required=True, help="Episode .npz files or directories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write per-step .npz samples.")
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=12)
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--rgb-key", type=str, default="rgb")
    parser.add_argument("--depth-key", type=str, default="depth")
    parser.add_argument("--goal-key", type=str, default="goal")
    parser.add_argument("--traj-key", type=str, default="trajectory")
    parser.add_argument("--value-key", type=str, default="value")
    parser.add_argument("--camera-pose-key", type=str, default="camera_pose")
    parser.add_argument("--local-points-key", type=str, default="local_points")
    parser.add_argument("--world-points-key", type=str, default="world_points")
    parser.add_argument("--allow-missing-trajectory", action="store_true")
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()


def discover_episode_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve(strict=False)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.npz")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No episode .npz files found.")
    return files


def load_episode(path: Path, args: argparse.Namespace) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        episode = {key: data[key] for key in data.files}

    for key in (args.rgb_key, args.depth_key, args.goal_key):
        if key not in episode:
            raise KeyError(f"{path} missing required key: {key}")
    if args.traj_key not in episode and not args.allow_missing_trajectory:
        raise KeyError(
            f"{path} missing required key: {args.traj_key}. "
            "Pass --allow-missing-trajectory to synthesize from goal deltas."
        )
    return episode


def ensure_rgb_shape(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected rgb shape (T,H,W,3), got {rgb.shape}")
    return rgb


def ensure_depth_shape(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        depth = depth[..., np.newaxis]
    if depth.ndim != 4 or depth.shape[-1] != 1:
        raise ValueError(f"Expected depth shape (T,H,W,1), got {depth.shape}")
    return depth


def normalize_goal(goal: np.ndarray) -> np.ndarray:
    if goal.ndim != 2:
        raise ValueError(f"Expected goal shape (T,D), got {goal.shape}")
    if goal.shape[1] == 2:
        padded = np.zeros((goal.shape[0], 3), dtype=np.float32)
        padded[:, :2] = goal.astype(np.float32)
        return padded
    if goal.shape[1] == 3:
        return goal.astype(np.float32)
    raise ValueError(f"Unsupported goal shape: {goal.shape}")


def resize_rgb_sequence(rgb: np.ndarray) -> np.ndarray:
    output = []
    for image in rgb:
        image_uint8 = image
        if image_uint8.dtype != np.uint8:
            if np.max(image_uint8) <= 1.0:
                image_uint8 = np.clip(image_uint8 * 255.0, 0, 255).astype(np.uint8)
            else:
                image_uint8 = np.clip(image_uint8, 0, 255).astype(np.uint8)
        resized = cv2.resize(image_uint8, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        output.append(resized.astype(np.float32) / 255.0)
    return np.asarray(output, dtype=np.float32)


def resize_depth_sequence(depth: np.ndarray) -> np.ndarray:
    output = []
    for image in depth:
        plane = image[..., 0].astype(np.float32)
        plane[~np.isfinite(plane)] = 0.0
        resized = cv2.resize(plane, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        output.append(resized[..., np.newaxis].astype(np.float32))
    return np.asarray(output, dtype=np.float32)


def clip_depth(rgbd: np.ndarray) -> np.ndarray:
    rgbd = rgbd.copy()
    depth = rgbd[..., 3]
    depth[(depth > 5.0) | (depth < 0.1)] = 0.0
    rgbd[..., 3] = depth
    return rgbd


def get_context_indices(current_index: int, context_size: int) -> list[int]:
    if current_index + 1 < context_size:
        indices = [0] * (context_size - current_index - 1)
        indices.extend(range(0, current_index + 1))
        return indices
    step = current_index / max(context_size - 1, 1)
    indices = [int(round(i * step)) for i in range(context_size)]
    indices[-1] = current_index
    return indices


def build_memory_window(rgb: np.ndarray, depth: np.ndarray, t: int, memory_size: int) -> np.ndarray:
    start_idx = max(t + 1 - memory_size, 0)
    indices = list(range(start_idx, t + 1))
    if len(indices) < memory_size:
        indices = [0] * (memory_size - len(indices)) + indices
    memory_rgb = rgb[indices]
    memory_depth = depth[indices]
    return clip_depth(np.concatenate([memory_rgb, memory_depth], axis=-1))


def build_context_window(rgb: np.ndarray, depth: np.ndarray, t: int, context_size: int) -> np.ndarray:
    indices = get_context_indices(t, context_size)
    context_rgb = rgb[indices]
    context_depth = depth[indices]
    return clip_depth(np.concatenate([context_rgb, context_depth], axis=-1))


def build_target_trajectory(goal: np.ndarray, t: int, predict_size: int) -> np.ndarray:
    target = np.zeros((predict_size, 3), dtype=np.float32)
    remaining = goal[t : t + predict_size]
    target[: remaining.shape[0]] = remaining[:, :3]
    if remaining.shape[0] > 0 and remaining.shape[0] < predict_size:
        target[remaining.shape[0] :] = remaining[-1, :3]
    return target


def get_optional_sequence(episode: dict[str, np.ndarray], key: str, expected_len: int) -> np.ndarray | None:
    value = episode.get(key)
    if value is None:
        return None
    value = np.asarray(value)
    if value.shape[0] != expected_len:
        raise ValueError(f"Key {key} has length {value.shape[0]}, expected {expected_len}")
    return value


def to_jsonable_summary(summary: dict[str, object]) -> str:
    return json.dumps(summary, ensure_ascii=True, sort_keys=True)


def main() -> None:
    args = parse_args()
    episode_files = discover_episode_files(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    for episode_path in episode_files:
        episode = load_episode(episode_path, args)
        rgb = resize_rgb_sequence(ensure_rgb_shape(np.asarray(episode[args.rgb_key])))
        depth = resize_depth_sequence(ensure_depth_shape(np.asarray(episode[args.depth_key])))
        goal = normalize_goal(np.asarray(episode[args.goal_key]))
        seq_len = rgb.shape[0]
        if depth.shape[0] != seq_len or goal.shape[0] != seq_len:
            raise ValueError(
                f"Length mismatch in {episode_path}: rgb={rgb.shape[0]} depth={depth.shape[0]} goal={goal.shape[0]}"
            )

        trajectory = get_optional_sequence(episode, args.traj_key, seq_len)
        value = get_optional_sequence(episode, args.value_key, seq_len)
        camera_pose = get_optional_sequence(episode, args.camera_pose_key, seq_len)
        local_points = get_optional_sequence(episode, args.local_points_key, seq_len)
        world_points = get_optional_sequence(episode, args.world_points_key, seq_len)

        episode_stem = f"{args.prefix}{episode_path.stem}"
        for t in range(0, seq_len, args.step_stride):
            sample = {
                "memory_rgbd": build_memory_window(rgb, depth, t, args.memory_size),
                "context_rgbd": build_context_window(rgb, depth, t, args.context_size),
                "start_goal": goal[t].astype(np.float32),
                "target_traj": (
                    np.asarray(trajectory[t], dtype=np.float32)
                    if trajectory is not None and np.asarray(trajectory[t]).ndim == 2
                    else build_target_trajectory(goal, t, args.predict_size)
                ),
            }
            if value is not None:
                sample["target_value"] = np.asarray(value[t], dtype=np.float32)
            if camera_pose is not None:
                sample["target_camera_pose"] = np.asarray(camera_pose[get_context_indices(t, args.context_size)], dtype=np.float32)
            if local_points is not None:
                sample["target_local_points"] = np.asarray(local_points[get_context_indices(t, args.context_size)], dtype=np.float32)
            if world_points is not None:
                sample["target_world_points"] = np.asarray(world_points[get_context_indices(t, args.context_size)], dtype=np.float32)

            sample_path = args.output_dir / f"{episode_stem}_step{t:05d}.npz"
            np.savez_compressed(sample_path, **sample)
            total_samples += 1

    print(
        to_jsonable_summary(
            {
                "episodes": len(episode_files),
                "samples": total_samples,
                "output_dir": str(args.output_dir),
                "memory_size": args.memory_size,
                "context_size": args.context_size,
                "predict_size": args.predict_size,
            }
        )
    )


if __name__ == "__main__":
    main()
