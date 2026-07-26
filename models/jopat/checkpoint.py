"""Checkpoint loading for JOPAT."""

import torch

__all__ = ["load_from_checkpoint"]


def load_from_checkpoint(model, checkpoint_path, strict=False):
    """Load shape-compatible parameters from a JOPAT checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_state = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()

    loaded_params = []
    new_params = []
    mismatched_keys = []
    for key, value in model_state.items():
        if key in checkpoint_state and value.shape == checkpoint_state[key].shape:
            model_state[key] = checkpoint_state[key]
            loaded_params.append(key)
        else:
            new_params.append(key)
            if key in checkpoint_state:
                mismatched_keys.append(key)
                print(
                    f"Shape mismatch for {key}: "
                    f"{tuple(checkpoint_state[key].shape)} -> {tuple(value.shape)}"
                )

    model.load_state_dict(model_state, strict=strict)
    loading_ratio = len(loaded_params) / max(len(model_state), 1)
    print(
        f"Loaded {len(loaded_params)} of {len(model_state)} JOPAT parameters "
        f"({100.0 * loading_ratio:.1f}%)."
    )

    load_info = {
        "loaded_params": loaded_params,
        "new_params": new_params,
        "mismatched_keys": mismatched_keys,
        "original_step": checkpoint.get("step", 0),
        "original_epoch": checkpoint.get("epoch", 0),
        "loading_ratio": loading_ratio,
    }
    return model, load_info
