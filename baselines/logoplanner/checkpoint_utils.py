from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import torch


def _maybe_get_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "module", "weights"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise TypeError("Unsupported checkpoint format. Expected a state-dict-like mapping.")


def _strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    prefixes = ("module.", "model.", "navi_former.")
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def load_checkpoint_payload(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=map_location)
    except pickle.UnpicklingError:
        # Local training checkpoints may contain non-tensor metadata such as
        # pathlib.Path objects. Fall back to the legacy loader for trusted
        # workspace checkpoints.
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_model_state_dict(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    payload = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    state_dict = _maybe_get_state_dict(payload)
    return _strip_known_prefixes(state_dict)


def load_weights_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    state_dict = load_model_state_dict(checkpoint_path, map_location=map_location)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
