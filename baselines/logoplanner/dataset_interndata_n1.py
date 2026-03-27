from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_H = 168
TARGET_W = 308


def discover_tar_files(inputs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser().resolve(strict=False)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.tar.gz")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No InternData-N1 tar.gz files found.")
    return files


def yaw_from_rotation(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def pitch_from_rotation(rotation: np.ndarray) -> float:
    return float(np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)))


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


def pose4_to_model_camera_target(poses: np.ndarray) -> np.ndarray:
    """
    Convert wheeled-camera poses [x, y, z, yaw] into the 5D target used by
    LoGoPlanner's custom camera head: [x, y, z, sin(yaw), cos(yaw)].
    """
    xyz = poses[..., :3].astype(np.float32)
    yaw = poses[..., 3].astype(np.float32)
    sin_yaw = np.sin(yaw)[..., None].astype(np.float32)
    cos_yaw = np.cos(yaw)[..., None].astype(np.float32)
    return np.concatenate([xyz, sin_yaw, cos_yaw], axis=-1).astype(np.float32)


def positions_to_model_actions(relative_positions: np.ndarray) -> np.ndarray:
    """
    Convert future relative poses into the action-like sequence expected by
    LoGoPlanner's diffusion head.

    In inference, the model predicts `naction` and then reconstructs a future
    trajectory as:

        all_trajectory = cumsum(naction / 4.0, dim=1)

    So the training target for the diffusion model should be the per-step
    increments in local frame, multiplied by 4.
    """
    deltas = np.empty_like(relative_positions, dtype=np.float32)
    deltas[0] = relative_positions[0]
    deltas[1:] = relative_positions[1:] - relative_positions[:-1]
    return deltas * 4.0


def get_context_indices(current_index: int, context_size: int) -> list[int]:
    if current_index + 1 < context_size:
        indices = [0] * (context_size - current_index - 1)
        indices.extend(range(0, current_index + 1))
        return indices
    step = current_index / max(context_size - 1, 1)
    indices = [int(round(i * step)) for i in range(context_size)]
    indices[-1] = current_index
    return indices


def resize_rgb(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32) / 255.0


def resize_depth(image: np.ndarray, depth_scale: float) -> np.ndarray:
    resized = cv2.resize(image, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / depth_scale)[..., np.newaxis]


def clip_depth(rgbd: np.ndarray) -> np.ndarray:
    rgbd = rgbd.copy()
    depth = rgbd[..., 3]
    depth[(depth > 5.0) | (depth < 0.1)] = 0.0
    rgbd[..., 3] = depth
    return rgbd


def depth_to_local_points(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """
    Unproject a depth map into camera-local XYZ coordinates.
    """
    h, w = depth.shape
    fu = intrinsic[0, 0]
    fv = intrinsic[1, 1]
    cu = intrinsic[0, 2]
    cv = intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float32)
    x = (u - cu) * z / max(fu, 1e-8)
    y = (v - cv) * z / max(fv, 1e-8)
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    points[z <= 0.0] = 0.0
    return points


def local_to_world_points(local_points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """
    Transform camera-local XYZ points into world coordinates using a 4x4
    camera-to-world extrinsic.
    """
    rotation = extrinsic[:3, :3].astype(np.float32)
    translation = extrinsic[:3, 3].astype(np.float32)
    world = np.einsum("ij,hwj->hwi", rotation, local_points) + translation[None, None, :]
    invalid = local_points[..., 2] <= 0.0
    world[invalid] = 0.0
    return world.astype(np.float32)


class InternDataN1TarDataset(Dataset):
    def __init__(
        self,
        inputs: Iterable[str],
        memory_size: int = 8,
        context_size: int = 12,
        predict_size: int = 24,
        depth_scale: float = 1000.0,
        step_stride: int = 1,
        max_episodes_per_tar: int = 0,
        episode_cache_dir: str | Path | None = None,
        target_camera_height: float | None = None,
        target_camera_pitch_deg: float | None = None,
        camera_height_tol: float = 0.05,
        camera_pitch_tol_deg: float = 3.0,
    ):
        self.tar_files = discover_tar_files(inputs)
        self.memory_size = memory_size
        self.context_size = context_size
        self.predict_size = predict_size
        self.depth_scale = depth_scale
        self.step_stride = step_stride
        self.max_episodes_per_tar = max_episodes_per_tar
        self.episode_cache_dir = Path(episode_cache_dir).expanduser().resolve(strict=False) if episode_cache_dir else None
        self.target_camera_height = target_camera_height
        self.target_camera_pitch_deg = target_camera_pitch_deg
        self.camera_height_tol = camera_height_tol
        self.camera_pitch_tol_deg = camera_pitch_tol_deg
        if self.episode_cache_dir is not None:
            self.episode_cache_dir.mkdir(parents=True, exist_ok=True)

        self.index: list[dict[str, object]] = []
        self._episode_cache: dict[tuple[str, int], dict[str, object]] = {}

        for tar_path in self.tar_files:
            with tarfile.open(tar_path, "r:gz") as archive:
                scene_root = None
                parquet_members = []
                stats = {}
                for member in archive.getmembers():
                    name = member.name
                    if scene_root is None:
                        scene_root = Path(name).parts[0]
                    if name.endswith(".parquet") and "/data/chunk-" in name:
                        parquet_members.append(name)
                    if name.endswith("meta/episodes_stats.jsonl"):
                        stats_bytes = self._read_member_by_name(archive, name)
                        stats = self._parse_episode_stats(stats_bytes)
                parquet_members = sorted(parquet_members)
                if self.max_episodes_per_tar > 0:
                    parquet_members = parquet_members[: self.max_episodes_per_tar]
                if scene_root is None:
                    continue
                for parquet_member in parquet_members:
                    episode_index = int(Path(parquet_member).stem.split("_")[-1])
                    seq_len = stats.get(episode_index)
                    parquet_bytes = None
                    df = None
                    if seq_len is None or self._use_camera_pose_filter():
                        parquet_bytes = self._read_member_by_name(archive, parquet_member)
                        df = pd.read_parquet(io.BytesIO(parquet_bytes))
                    if seq_len is None:
                        assert df is not None
                        seq_len = len(df)
                    valid_step_indices = range(0, seq_len, self.step_stride)
                    if self._use_camera_pose_filter():
                        assert df is not None
                        extrinsics = np.stack(df["observation.camera_extrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 4, 4)
                        valid_step_indices = [
                            step_index
                            for step_index in range(0, seq_len, self.step_stride)
                            if self._camera_pose_matches_filter(extrinsics[step_index])
                        ]
                    for step_index in valid_step_indices:
                        self.index.append(
                            {
                                "tar_path": str(tar_path),
                                "scene_root": scene_root,
                                "episode_index": episode_index,
                                "step_index": step_index,
                            }
                        )
        if not self.index:
            raise ValueError("No training samples remain after camera pose filtering.")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        item = self.index[index]
        tar_path = str(item["tar_path"])
        episode_index = int(item["episode_index"])
        step_index = int(item["step_index"])
        episode = self._load_episode(tar_path=tar_path, scene_root=str(item["scene_root"]), episode_index=episode_index)

        rgb = episode["rgb"]
        depth = episode["depth"]
        poses = episode["poses"]
        intrinsics = episode["intrinsics"]
        extrinsics = episode["extrinsics"]

        memory_indices = list(range(max(step_index + 1 - self.memory_size, 0), step_index + 1))
        if len(memory_indices) < self.memory_size:
            memory_indices = [0] * (self.memory_size - len(memory_indices)) + memory_indices
        context_indices = get_context_indices(step_index, self.context_size)

        memory_rgbd = clip_depth(np.concatenate([rgb[memory_indices], depth[memory_indices]], axis=-1))
        context_rgbd = clip_depth(np.concatenate([rgb[context_indices], depth[context_indices]], axis=-1))

        start_goal = relative_pose_2d(poses[step_index], poses[-1])
        target_positions = np.stack(
            [
                relative_pose_2d(poses[step_index], poses[min(step_index + offset + 1, len(poses) - 1)])
                for offset in range(self.predict_size)
            ],
            axis=0,
        ).astype(np.float32)
        target_traj = positions_to_model_actions(target_positions)
        target_camera_pose = pose4_to_model_camera_target(poses[context_indices])
        target_local_points = np.stack(
            [depth_to_local_points(depth[idx, ..., 0], intrinsics[idx]) for idx in context_indices],
            axis=0,
        ).astype(np.float32)
        target_world_points = np.stack(
            [local_to_world_points(target_local_points[i], extrinsics[idx]) for i, idx in enumerate(context_indices)],
            axis=0,
        ).astype(np.float32)

        return {
            "memory_rgbd": torch.from_numpy(memory_rgbd).float(),
            "context_rgbd": torch.from_numpy(context_rgbd).float(),
            "start_goal": torch.from_numpy(start_goal).float(),
            "target_traj": torch.from_numpy(target_traj).float(),
            "target_camera_pose": torch.from_numpy(target_camera_pose).float(),
            "target_local_points": torch.from_numpy(target_local_points).float(),
            "target_world_points": torch.from_numpy(target_world_points).float(),
            "camera_intrinsic": torch.from_numpy(intrinsics[step_index]).float(),
            "sample_path": f"{tar_path}::episode_{episode_index:06d}::step_{step_index:06d}",
        }

    def _load_episode(self, tar_path: str, scene_root: str, episode_index: int) -> dict[str, object]:
        cache_key = (tar_path, episode_index)
        if cache_key in self._episode_cache:
            return self._episode_cache[cache_key]

        cache_path = self._episode_cache_path(tar_path, episode_index)
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as data:
                episode = {
                    "rgb": data["rgb"].astype(np.float32),
                    "depth": data["depth"].astype(np.float32),
                    "poses": data["poses"].astype(np.float32),
                    "intrinsics": data["intrinsics"].astype(np.float32),
                    "extrinsics": data["extrinsics"].astype(np.float32),
                }
            self._episode_cache[cache_key] = episode
            return episode

        with tarfile.open(tar_path, "r:gz") as archive:
            parquet_name = f"{scene_root}/data/chunk-000/episode_{episode_index:06d}.parquet"
            parquet_bytes = self._read_member_by_name(archive, parquet_name)
            df = pd.read_parquet(io.BytesIO(parquet_bytes))
            seq_len = len(df)

            extrinsics = np.stack(df["observation.camera_extrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 4, 4)
            intrinsics = np.stack(df["observation.camera_intrinsic"].to_list(), axis=0).astype(np.float32).reshape(seq_len, 3, 3)
            poses = np.stack([pose_summary(extrinsics[i]) for i in range(seq_len)], axis=0)

            rgb_frames = []
            depth_frames = []
            for frame_index in range(seq_len):
                rgb_name = f"{scene_root}/videos/chunk-000/observation.images.rgb/episode_{episode_index:06d}_{frame_index:03d}.jpg"
                depth_name = f"{scene_root}/videos/chunk-000/observation.images.depth/episode_{episode_index:06d}_{frame_index:03d}.png"
                rgb_frames.append(resize_rgb(self._decode_rgb(self._read_member_by_name(archive, rgb_name))))
                depth_frames.append(resize_depth(self._decode_depth(self._read_member_by_name(archive, depth_name)), self.depth_scale))

        episode = {
            "rgb": np.stack(rgb_frames, axis=0).astype(np.float32),
            "depth": np.stack(depth_frames, axis=0).astype(np.float32),
            "poses": poses.astype(np.float32),
            "intrinsics": intrinsics.astype(np.float32),
            "extrinsics": extrinsics.astype(np.float32),
        }
        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                rgb=episode["rgb"],
                depth=episode["depth"],
                poses=episode["poses"],
                intrinsics=episode["intrinsics"],
                extrinsics=episode["extrinsics"],
            )
        self._episode_cache[cache_key] = episode
        return episode

    def _episode_cache_path(self, tar_path: str, episode_index: int) -> Path | None:
        if self.episode_cache_dir is None:
            return None
        tar_stem = Path(tar_path).name.replace(".tar.gz", "")
        return self.episode_cache_dir / f"{tar_stem}_episode_{episode_index:06d}.npz"

    def _use_camera_pose_filter(self) -> bool:
        return self.target_camera_height is not None or self.target_camera_pitch_deg is not None

    def _camera_pose_matches_filter(self, extrinsic: np.ndarray) -> bool:
        translation = extrinsic[:3, 3]
        rotation = extrinsic[:3, :3]
        if self.target_camera_height is not None:
            if abs(float(translation[2]) - self.target_camera_height) > self.camera_height_tol:
                return False
        if self.target_camera_pitch_deg is not None:
            pitch_deg = float(np.degrees(pitch_from_rotation(rotation)))
            if abs(pitch_deg - self.target_camera_pitch_deg) > self.camera_pitch_tol_deg:
                return False
        return True

    @staticmethod
    def _parse_episode_stats(stats_bytes: bytes) -> dict[int, int]:
        stats = {}
        for raw_line in stats_bytes.decode("utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            episode_index = int(record["episode_index"])
            image_index = record.get("image_index", {})
            count = int(image_index.get("count", 0))
            if count > 0:
                stats[episode_index] = count
        return stats

    @staticmethod
    def _read_member_by_name(archive: tarfile.TarFile, member_name: str) -> bytes:
        member = archive.getmember(member_name)
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Archive member not found: {member_name}")
        try:
            return extracted.read()
        finally:
            extracted.close()

    @staticmethod
    def _decode_rgb(image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode RGB frame.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _decode_depth(image_bytes: bytes) -> np.ndarray:
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to decode depth frame.")
        return image
