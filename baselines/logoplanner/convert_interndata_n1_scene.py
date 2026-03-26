from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an extracted InternData-N1 scene directory into per-episode LoGoPlanner .npz files."
    )
    parser.add_argument("--scene-dir", type=Path, required=True, help="Extracted scene directory from a traj_data tar.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write episode .npz files.")
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Divide decoded depth PNG values by this number to convert to meters.",
    )
    return parser.parse_args()


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def pose_summary(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    yaw = yaw_from_rotation(rotation)
    return np.asarray([translation[0], translation[1], translation[2], yaw], dtype=np.float32)


def relative_pose_2d(origin: np.ndarray, target: np.ndarray) -> np.ndarray:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    cos_yaw = np.cos(origin[3])
    sin_yaw = np.sin(origin[3])
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    local_yaw = target[3] - origin[3]
    return np.asarray([local_x, local_y, local_yaw], dtype=np.float32)


def load_rgb_frame(rgb_dir: Path, episode_index: int, frame_index: int) -> np.ndarray:
    path = rgb_dir / f"episode_{episode_index:06d}_{frame_index:03d}.jpg"
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"RGB frame not found: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_depth_frame(depth_dir: Path, episode_index: int, frame_index: int, depth_scale: float) -> np.ndarray:
    path = depth_dir / f"episode_{episode_index:06d}_{frame_index:03d}.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Depth frame not found: {path}")
    image = image.astype(np.float32) / depth_scale
    return image[..., np.newaxis]


def build_future_trajectory(poses: np.ndarray, start_index: int, horizon: int) -> np.ndarray:
    trajectory = np.zeros((horizon, 3), dtype=np.float32)
    origin = poses[start_index]
    for step in range(horizon):
        future_index = min(start_index + step, len(poses) - 1)
        trajectory[step] = relative_pose_2d(origin, poses[future_index])
    return trajectory


def main() -> None:
    args = parse_args()
    data_dir = args.scene_dir / "data" / "chunk-000"
    rgb_dir = args.scene_dir / "videos" / "chunk-000" / "observation.images.rgb"
    depth_dir = args.scene_dir / "videos" / "chunk-000" / "observation.images.depth"

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB image directory: {rgb_dir}")
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Missing depth image directory: {depth_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found in {data_dir}")

    converted = 0
    for parquet_path in parquet_files:
        episode_index = int(parquet_path.stem.split("_")[-1])
        df = pd.read_parquet(parquet_path)
        seq_len = len(df)
        if seq_len == 0:
            continue

        extrinsics = np.stack(df["observation.camera_extrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 4, 4)
        intrinsics = np.stack(df["observation.camera_intrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 3, 3)
        poses = np.stack([pose_summary(extrinsics[i]) for i in range(seq_len)], axis=0)

        rgb_frames = np.stack(
            [load_rgb_frame(rgb_dir, episode_index, frame_index) for frame_index in range(seq_len)],
            axis=0,
        ).astype(np.uint8)
        depth_frames = np.stack(
            [load_depth_frame(depth_dir, episode_index, frame_index, args.depth_scale) for frame_index in range(seq_len)],
            axis=0,
        ).astype(np.float32)

        goal = np.stack(
            [relative_pose_2d(poses[t], poses[-1]) for t in range(seq_len)],
            axis=0,
        ).astype(np.float32)
        trajectory = np.stack(
            [build_future_trajectory(poses, t, args.predict_size) for t in range(seq_len)],
            axis=0,
        ).astype(np.float32)

        output_path = args.output_dir / f"episode_{episode_index:06d}.npz"
        np.savez_compressed(
            output_path,
            rgb=rgb_frames,
            depth=depth_frames,
            goal=goal,
            trajectory=trajectory,
            camera_pose=poses,
            camera_intrinsic=intrinsics,
        )
        converted += 1
        print(f"converted {parquet_path.name} -> {output_path}")

    print(f"converted_episodes={converted}")


if __name__ == "__main__":
    main()
