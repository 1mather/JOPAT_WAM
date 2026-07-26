import os
import random
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


_WANDB = None


def _infer_default_data_type() -> torch.dtype:
    try:
        if not torch.cuda.is_available():
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except (RuntimeError, AttributeError):
        return torch.float32


DATA_TYPE = _infer_default_data_type()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wandb_is_enabled(config: Any) -> bool:
    enabled = OmegaConf.select(config, "wandb.enabled", default=None)
    if enabled is None:
        enabled = OmegaConf.select(config, "use_wandb", default=False)
    return bool(enabled)


def init_wandb(config: Any, job_type: str):
    global _WANDB

    if not wandb_is_enabled(config):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled, but the 'wandb' package is not installed."
        ) from exc

    init_args = {
        "job_type": job_type,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    for key in ("project", "entity", "group", "name", "mode", "tags"):
        value = OmegaConf.select(config, f"wandb.{key}", default=None)
        if value is not None:
            init_args[key] = value

    run_id = OmegaConf.select(config, "wandb.run_id", default=None)
    if run_id is not None:
        init_args["id"] = run_id
        init_args["resume"] = "allow" if bool(config.get("resume", False)) else None

    _WANDB = wandb
    return wandb.init(**init_args)


def get_wandb():
    return _WANDB


def init_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25678")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=3600),
    )


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def distributed_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def soft_update(target, source, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


class FreezeParameters:
    def __init__(self, params):
        self.params = list(params)
        self.param_states = [parameter.requires_grad for parameter in self.params]

    def __enter__(self):
        for parameter in self.params:
            parameter.requires_grad = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for parameter, state in zip(self.params, self.param_states):
            parameter.requires_grad = state
