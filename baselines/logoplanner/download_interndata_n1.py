from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DATASET_ID = "InternRobotics/InternData-N1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a filtered subset of InternData-N1 from Hugging Face.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("huggingface_data/interndata_n1"),
        help="Local directory to place downloaded files.",
    )
    parser.add_argument(
        "--prefix",
        nargs="+",
        default=["vln_n1/traj_data/3dfront_d435i/"],
        help="One or more path prefixes to match inside the dataset repo.",
    )
    parser.add_argument("--max-files", type=int, default=1, help="Maximum number of matched files to download.")
    parser.add_argument(
        "--include-readme",
        action="store_true",
        help="Also download README.md into the output directory.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print matching files; do not download them.",
    )
    return parser.parse_args()


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN is required in the environment.")

    args = parse_args()
    api = HfApi(token=token)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    matches = [path for path in files if any(path.startswith(prefix) for prefix in args.prefix)]
    matches = sorted(matches)[: args.max_files]

    if args.include_readme:
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename="README.md",
            token=token,
            local_dir=args.output_dir,
        )

    if not matches:
        print("No files matched the given prefixes.")
        return

    for path in matches:
        print(path)
        if args.list_only:
            continue
        local_path = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=path,
            token=token,
            local_dir=args.output_dir,
        )
        print(f"downloaded -> {local_path}")


if __name__ == "__main__":
    main()
