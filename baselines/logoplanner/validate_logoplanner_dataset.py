from __future__ import annotations

import argparse
import json

import torch

from dataset_interndata_n1 import InternDataN1TarDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate LoGoPlanner samples generated on the fly from InternData-N1 tar files.")
    parser.add_argument("--data", nargs="+", required=True, help="InternData-N1 .tar.gz files or directories.")
    parser.add_argument("--max-samples", type=int, default=0, help="Validate only the first N samples. 0 means all.")
    parser.add_argument("--memory-size", type=int, default=8)
    parser.add_argument("--context-size", type=int, default=12)
    parser.add_argument("--predict-size", type=int, default=24)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--max-episodes-per-tar", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument(
        "--episode-cache-dir",
        type=str,
        default="",
        help="Optional directory for cached per-episode decoded InternData-N1 npz files.",
    )
    return parser.parse_args()


def summarize_tensor(name: str, value: torch.Tensor) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min().item()) if value.numel() > 0 else 0.0,
        "max": float(value.max().item()) if value.numel() > 0 else 0.0,
    }


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
    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)

    required_shapes: dict[str, tuple[int, ...]] | None = None
    inspected: list[dict[str, object]] = []
    for index in range(limit):
        sample = dataset[index]
        sample_shapes = {
            "memory_rgbd": tuple(sample["memory_rgbd"].shape),
            "context_rgbd": tuple(sample["context_rgbd"].shape),
            "start_goal": tuple(sample["start_goal"].shape),
            "target_traj": tuple(sample["target_traj"].shape),
        }
        if required_shapes is None:
            required_shapes = sample_shapes
        else:
            for key, shape in sample_shapes.items():
                if required_shapes[key] != shape:
                    raise ValueError(
                        f"Inconsistent shape for {key} at sample {index}: {shape} vs {required_shapes[key]}"
                    )

        record = {"sample_path": sample["sample_path"], "tensors": []}
        for key, value in sample.items():
            if torch.is_tensor(value):
                record["tensors"].append(summarize_tensor(key, value))
        inspected.append(record)

    print(
        json.dumps(
            {
                "num_files": len(dataset),
                "validated": limit,
                "reference_shapes": {key: list(value) for key, value in (required_shapes or {}).items()},
                "samples": inspected[: min(3, len(inspected))],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
