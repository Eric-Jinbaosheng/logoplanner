import json
import os
import sys
import time

import numpy as np
import torch


def main():
    device = os.environ.get("SMOKE_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    height = int(os.environ.get("SMOKE_HEIGHT", "28"))
    width = int(os.environ.get("SMOKE_WIDTH", "28"))
    context_size = int(os.environ.get("SMOKE_CONTEXT", "12"))
    output_path = os.environ.get("SMOKE_OUTPUT", "")

    sys.path.insert(0, os.path.dirname(__file__))
    from geometry_model import GeometryModel

    torch.manual_seed(123)
    np.random.seed(123)

    model = GeometryModel(context_size=context_size, device=device).to(device).eval()
    imgs = np.random.rand(1, context_size, height, width, 3).astype("float32")
    depths = np.random.rand(1, context_size, height, width, 1).astype("float32") * 5.0

    with torch.no_grad():
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        _ = model(imgs, depths)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        state_out, geom_out = model(imgs, depths)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    result = {
        "tag": "lact_only",
        "device": device,
        "height": height,
        "width": width,
        "context_size": context_size,
        "time_s": dt,
        "state_token_mean_abs": float(state_out[1].abs().mean().item()),
        "scene_token_mean_abs": float(state_out[2].abs().mean().item()),
        "camera_pose_mean_abs": float(geom_out[0].abs().mean().item()),
        "local_points_mean_abs": float(geom_out[1].abs().mean().item()),
        "world_points_mean_abs": float(geom_out[2].abs().mean().item()),
        "camera_shape": tuple(geom_out[0].shape),
        "local_shape": tuple(geom_out[1].shape),
    }
    print(json.dumps(result, indent=2))
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
