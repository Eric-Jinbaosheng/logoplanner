from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an InternData-N1 traj_data tar.gz directly into per-episode LoGoPlanner .npz files."
    )
    parser.add_argument("--tar-path", type=Path, required=True, help="Path to a traj_data scene tar.gz file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write episode .npz files.")
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-episodes", type=int, default=0, help="Limit conversion to first N episodes. 0 means all.")
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


def build_future_trajectory(poses: np.ndarray, start_index: int, horizon: int) -> np.ndarray:
    trajectory = np.zeros((horizon, 3), dtype=np.float32)
    origin = poses[start_index]
    for step in range(horizon):
        future_index = min(start_index + step, len(poses) - 1)
        trajectory[step] = relative_pose_2d(origin, poses[future_index])
    return trajectory


def read_member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(f"Archive member not found: {member.name}")
    try:
        return extracted.read()
    finally:
        extracted.close()


def decode_rgb(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode RGB image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_depth(image_bytes: bytes, depth_scale: float) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode depth image.")
    return (image.astype(np.float32) / depth_scale)[..., np.newaxis]


def main() -> None:
    args = parse_args()
    if not args.tar_path.is_file():
        raise FileNotFoundError(f"Tar file not found: {args.tar_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.tar_path, "r:gz") as archive:
        scene_root = None
        parquet_members: list[str] = []
        member_map: dict[str, tarfile.TarInfo] = {}
        for member in archive.getmembers():
            name = member.name
            member_map[name] = member
            parts = Path(name).parts
            if len(parts) >= 1 and scene_root is None:
                scene_root = parts[0]
            if name.endswith(".parquet") and "/data/chunk-" in name:
                parquet_members.append(name)

        if scene_root is None or not parquet_members:
            raise ValueError(f"No parquet episode data found in {args.tar_path}")

        parquet_members = sorted(parquet_members)
        if args.max_episodes > 0:
            parquet_members = parquet_members[: args.max_episodes]

        converted = 0
        for parquet_member in parquet_members:
            episode_index = int(Path(parquet_member).stem.split("_")[-1])
            parquet_bytes = read_member_bytes(archive, member_map[parquet_member])
            df = pd.read_parquet(io.BytesIO(parquet_bytes))
            seq_len = len(df)
            if seq_len == 0:
                continue

            extrinsics = np.stack(df["observation.camera_extrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 4, 4)
            intrinsics = np.stack(df["observation.camera_intrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 3, 3)
            poses = np.stack([pose_summary(extrinsics[i]) for i in range(seq_len)], axis=0)

            rgb_frames = []
            depth_frames = []
            for frame_index in range(seq_len):
                rgb_member = (
                    f"{scene_root}/videos/chunk-000/observation.images.rgb/"
                    f"episode_{episode_index:06d}_{frame_index:03d}.jpg"
                )
                depth_member = (
                    f"{scene_root}/videos/chunk-000/observation.images.depth/"
                    f"episode_{episode_index:06d}_{frame_index:03d}.png"
                )
                rgb_frames.append(decode_rgb(read_member_bytes(archive, member_map[rgb_member])))
                depth_frames.append(decode_depth(read_member_bytes(archive, member_map[depth_member]), args.depth_scale))

            rgb = np.stack(rgb_frames, axis=0).astype(np.uint8)
            depth = np.stack(depth_frames, axis=0).astype(np.float32)
            goal = np.stack([relative_pose_2d(poses[t], poses[-1]) for t in range(seq_len)], axis=0).astype(np.float32)
            trajectory = np.stack(
                [build_future_trajectory(poses, t, args.predict_size) for t in range(seq_len)],
                axis=0,
            ).astype(np.float32)

            output_path = args.output_dir / f"{args.tar_path.stem}_episode_{episode_index:06d}.npz"
            np.savez_compressed(
                output_path,
                rgb=rgb,
                depth=depth,
                goal=goal,
                trajectory=trajectory,
                camera_pose=poses,
                camera_intrinsic=intrinsics,
            )
            converted += 1
            print(f"converted episode {episode_index:06d} -> {output_path}")

    print(f"converted_episodes={converted}")


if __name__ == "__main__":
    main()
