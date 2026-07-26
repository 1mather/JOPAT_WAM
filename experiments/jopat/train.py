"""Distributed training, evaluation, and rollout entrypoint for JOPAT."""

import glob
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from datasets.utils.loader import make_distributed_data_loader
from environments.robomimic import make_robomimic_env
from experiments.utils import (
    distributed_barrier,
    get_wandb,
    init_distributed,
    init_wandb,
    is_main_process,
    set_seed,
)
from models.jopat.checkpoint import load_from_checkpoint


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "module", model)


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        if is_main_process():
            print(f"{name}: {time.perf_counter() - start:.2f}s")


def to_scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def format_metrics(metrics: Dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value:.4f}"
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.number))
    )


def safe_log_wandb(
    metrics: Dict[str, Any], step: int, prefix: str = ""
) -> None:
    wandb = get_wandb()
    if not is_main_process() or wandb is None:
        return

    payload = {
        f"{prefix}{key}" if prefix else key: value
        for key, value in metrics.items()
        if value is not None
    }
    if not payload:
        return

    try:
        wandb.log(payload, step=step)
    except Exception as exc:
        print(f"W&B logging failed: {exc}")


def as_wandb_image(image: torch.Tensor):
    wandb = get_wandb()
    return None if wandb is None else wandb.Image(image.detach().cpu())


def as_wandb_video(video: np.ndarray, fps: int = 10):
    wandb = get_wandb()
    if wandb is None:
        return None
    frames = np.asarray(video)
    frames = np.transpose(frames, (0, 3, 1, 2))
    return wandb.Video(frames, fps=fps, format="mp4")


def process_batch(
    batch: Dict[str, Any],
    obs_horizon: int,
    action_horizon: int,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    action_start = obs_horizon - 1
    action_end = action_start + action_horizon
    curr_obs = {
        key: value[:, : action_start + 1].to(device, non_blocking=True)
        for key, value in batch["obs"].items()
    }
    next_obs = {
        key: value[:, action_end:].to(device, non_blocking=True)
        for key, value in batch["obs"].items()
    }
    action = batch["action"][:, action_start:action_end].to(
        device, non_blocking=True
    )

    for key in ("input_ids", "attention_mask"):
        if key in batch:
            tokens = batch[key]
            if tokens.ndim == 3:
                tokens = tokens[:, 0]
            curr_obs[key] = tokens.to(device, non_blocking=True)
    return curr_obs, next_obs, action


def get_image_size(batch: Dict[str, Any], config: DictConfig) -> Tuple[int, int]:
    for key, metadata in config.dataset.shape_meta.obs.items():
        if metadata.get("type") != "rgb" or key not in batch["obs"]:
            continue
        images = batch["obs"][key]
        if images.ndim != 5:
            raise ValueError(
                f"Expected RGB observations with shape (B, T, H, W, C), got "
                f"{tuple(images.shape)} for {key!r}."
            )
        return int(images.shape[-3]), int(images.shape[-2])
    raise KeyError("The batch does not contain an RGB observation from shape_meta.")


def process_tracks(
    batch: Dict[str, Any],
    device: torch.device,
    height: int,
    width: int,
    max_tracks: int,
    border_fraction: float = 10.0 / 128.0,
) -> Optional[torch.Tensor]:
    tracks = batch.get("track")
    if tracks is None:
        return None

    tracks = tracks.to(device, non_blocking=True)
    batch_size, num_frames, num_tracks, num_dims = tracks.shape
    coordinate_scale = torch.tensor(
        [width, height], dtype=tracks.dtype, device=device
    )
    normalized = tracks / coordinate_scale * 2.0 - 1.0

    lower = -1.0 + 2.0 * border_fraction
    upper = 1.0 - 2.0 * border_fraction
    valid = ((normalized >= lower) & (normalized <= upper)).all(dim=-1)
    valid = valid.all(dim=1)
    movement = normalized.std(dim=1).mean(dim=-1)
    movement = movement.masked_fill(~valid, float("-inf"))

    count = min(max_tracks, num_tracks)
    scores, indices = torch.topk(movement, k=count, dim=1)
    if is_main_process() and torch.any(torch.isneginf(scores)):
        print("Warning: fewer valid tracks than requested; using fallback tracks.")

    gather_indices = indices[:, None, :, None].expand(
        batch_size, num_frames, count, num_dims
    )
    selected = torch.gather(normalized, dim=2, index=gather_indices)

    visibility = batch.get("track_visibility")
    if visibility is not None:
        visibility = visibility.to(device, non_blocking=True)
        visibility_indices = indices[:, None, :].expand(
            batch_size, num_frames, count
        )
        selected_visibility = torch.gather(
            visibility, dim=2, index=visibility_indices
        )
        selected = torch.cat(
            (selected, selected_visibility.unsqueeze(-1)), dim=-1
        )
    return selected


def train_one_step(
    config: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    batch: Dict[str, Any],
    device: torch.device,
    step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.train()
    curr_obs, next_obs, action = process_batch(
        batch,
        config.model.obs_encoder.num_frames,
        config.model.action_len,
        device,
    )
    height, width = get_image_size(batch, config)
    max_tracks = int(config.model.get("num_points", 625))
    tracks = process_tracks(batch, device, height, width, max_tracks)
    action_mask = batch.get("action_mask")
    if action_mask is not None:
        action_mask = action_mask.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=bool(config.use_amp),
    ):
        loss, info = model(
            obs_dict=curr_obs,
            next_obs_dict=next_obs,
            action=action,
            tracks=tracks,
            action_mask=action_mask,
        )

    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

    scaler.scale(loss).backward()
    clip_norm = config.get("clip_grad_norm")
    if clip_norm:
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        info["grad_norm"] = to_scalar(grad_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    info = {key: to_scalar(value) for key, value in info.items()}
    info["lr"] = float(scheduler.get_last_lr()[0])
    info["step"] = float(step)
    return loss, info


def maybe_resume_checkpoint(
    config: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
) -> int:
    if not bool(config.get("resume", False)):
        return 0

    checkpoint_path = config.get("resume_checkpoint_path")
    if not checkpoint_path:
        raise ValueError(
            "resume_checkpoint_path must be set when resume is enabled."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    step = int(checkpoint["step"]) + 1
    if is_main_process():
        print(f"Resumed from {checkpoint_path} at step {step}.")
    return step


def maybe_save_checkpoint(
    config: DictConfig,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
) -> Optional[Path]:
    save_every = int(config.save_every)
    should_save = step % save_every == 0 or step == int(config.num_steps) - 1
    if not is_main_process() or not should_save:
        return None

    checkpoint_dir = Path(config.logdir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
    temporary_path = checkpoint_path.with_suffix(".tmp")
    checkpoint = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}.")
    return checkpoint_path


def decode_and_plot(
    model: nn.Module, latents: torch.Tensor, nrows: int = 4
):
    with torch.no_grad():
        images = model.obs_encoder.apply_vae(latents, inverse=True)
        images = rearrange(images, "b v c t h w -> (b v t) c h w")
        return as_wandb_image(make_grid(images, nrow=nrows))


def visualize_tracks_on_frame(
    image: np.ndarray,
    tracks: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    num_points: int = 50,
    point_radius: int = 2,
) -> np.ndarray:
    output = np.asarray(image).copy()
    height, width = output.shape[:2]
    if len(tracks) > num_points:
        indices = np.linspace(0, len(tracks) - 1, num_points, dtype=int)
        tracks = tracks[indices]
        if visibility is not None:
            visibility = visibility[indices]

    points = np.asarray(tracks).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    points = points.astype(np.int32)
    for index, (x_coord, y_coord) in enumerate(points):
        if not (0 <= x_coord < width and 0 <= y_coord < height):
            continue
        score = 1.0 if visibility is None else float(visibility[index])
        color = np.array(
            [int(255 * (1.0 - score)), int(255 * score), 0],
            dtype=output.dtype,
        )
        x_start = max(0, x_coord - point_radius)
        x_end = min(width, x_coord + point_radius + 1)
        y_start = max(0, y_coord - point_radius)
        y_end = min(height, y_coord + point_radius + 1)
        output[y_start:y_end, x_start:x_end] = color
    return output


def get_rollout_dataset_path(config: DictConfig) -> Optional[str]:
    if not bool(config.get("enable_rollouts", True)):
        return None
    if not config.dataset.get("rollout_name"):
        return None

    dataset_path = config.dataset.get("hdf5_path_globs")
    if not dataset_path:
        return None
    dataset_path = str(dataset_path)
    if not glob.has_magic(dataset_path):
        return dataset_path

    matches = sorted(glob.glob(dataset_path))
    return matches[0] if len(matches) == 1 else None


def collect_rollouts(
    config: DictConfig,
    model: nn.Module,
    device: torch.device,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    model = unwrap_model(model)
    model.eval()
    successes = []
    last_video = None
    last_annotated_video = None
    dataset_path = get_rollout_dataset_path(config)
    if dataset_path is None:
        raise ValueError(
            "Rollouts require one dataset file and a dataset.rollout_name."
        )

    for episode in trange(
        int(config.num_rollouts),
        desc="Rollouts",
        disable=not is_main_process(),
    ):
        env = make_robomimic_env(
            dataset_name=config.dataset.rollout_name,
            dataset_path=dataset_path,
            shape_meta=config.dataset.shape_meta,
            obs_horizon=model.obs_encoder.num_frames,
            max_episode_length=config.rollout_length,
            record=True,
        )
        track_chunks = []
        visibility_chunks = []
        last_annotated_video = None
        info = {}
        try:
            env.seed(int(config.seed) + episode)
            observation = env.reset()
            done = False
            while not done:
                observation_tensor = {
                    key: torch.as_tensor(value, device=device)[None]
                    for key, value in observation.items()
                }
                if hasattr(model, "sample_action_and_tracks"):
                    action, tracks, visibility = model.sample_action_and_tracks(
                        observation_tensor
                    )
                    action = action[0].detach().cpu().numpy()
                    track_chunk = tracks[0].detach().cpu().numpy()
                    chunk_length = min(len(action), max(0, len(track_chunk) - 1))
                    track_chunks.append(track_chunk[1 : chunk_length + 1])
                    if visibility is not None:
                        visibility_chunk = visibility[0].detach().cpu().numpy()
                        visibility_chunks.append(
                            visibility_chunk[1 : chunk_length + 1]
                        )
                else:
                    action = model.sample(observation_tensor)[0]
                    action = action.detach().cpu().numpy()
                observation, _, done, info = env.step(action)

            successes.append(bool(info.get("success", False)))
            try:
                last_video = np.asarray(env.get_video())
            except (AttributeError, RuntimeError):
                last_video = None
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()

        if last_video is not None and track_chunks:
            flattened_tracks = np.concatenate(track_chunks, axis=0)
            normalized_tracks = np.clip(
                (flattened_tracks + 1.0) / 2.0, 0.0, 1.0
            )
            flattened_visibility = (
                np.concatenate(visibility_chunks, axis=0)
                if visibility_chunks
                else None
            )
            last_annotated_video = last_video.copy()
            frame_count = min(len(last_annotated_video), len(normalized_tracks))
            for frame_index in range(frame_count):
                frame_visibility = (
                    None
                    if flattened_visibility is None
                    else flattened_visibility[frame_index]
                )
                last_annotated_video[frame_index] = visualize_tracks_on_frame(
                    last_annotated_video[frame_index],
                    normalized_tracks[frame_index],
                    frame_visibility,
                )

    success_rate = float(np.mean(successes)) if successes else 0.0
    return success_rate, last_video, last_annotated_video


def rollout_is_supported(config: DictConfig) -> bool:
    return get_rollout_dataset_path(config) is not None


def maybe_collect_rollout(
    config: DictConfig,
    step: int,
    model: nn.Module,
    device: torch.device,
) -> None:
    if not rollout_is_supported(config):
        return

    should_collect = (
        step % int(config.rollout_every) == 0
        or step == int(config.num_steps) - 1
    )
    if not should_collect:
        return

    if is_main_process():
        try:
            print(f"Step {step}: collecting rollouts.")
            with timer("Rollout collection"):
                success_rate, video, annotated_video = collect_rollouts(
                    config, model, device
                )
            metrics = {"success_rate": success_rate}
            if video is not None:
                metrics["video"] = as_wandb_video(video)
            if annotated_video is not None:
                metrics["video_with_tracks"] = as_wandb_video(annotated_video)
            safe_log_wandb(metrics, step, prefix="rollout/")
            print(f"Rollout success rate: {success_rate:.2%}")
        except Exception as exc:
            print(f"Rollout collection failed at step {step}: {exc}")
    distributed_barrier()


def reduce_metrics(
    metric_sums: Dict[str, float],
    num_batches: int,
    device: torch.device,
) -> Dict[str, float]:
    count = torch.tensor(float(num_batches), device=device)
    if dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if count.item() == 0:
        raise RuntimeError("The validation loader produced no batches.")

    reduced = {}
    for key, value in metric_sums.items():
        total = torch.tensor(float(value), device=device)
        if dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        reduced[key] = (total / count).item()
    return reduced


def evaluate(
    config: DictConfig,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    action_normalizer: Optional[Any] = None,
) -> Dict[str, Any]:
    model = unwrap_model(model)
    model.eval()
    if action_normalizer is None:
        unnormalize = lambda value: value
    else:
        action_scale = torch.as_tensor(
            action_normalizer.scale[None], device=device
        )
        action_offset = torch.as_tensor(
            action_normalizer.offset[None], device=device
        )
        unnormalize = lambda value: value * action_scale + action_offset

    metric_sums = defaultdict(float)
    num_batches = 0
    visualizations = None
    max_batches = config.get("max_eval_batches")

    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(
                val_loader,
                desc="Validation",
                disable=not is_main_process(),
            )
        ):
            if max_batches is not None and batch_index >= int(max_batches):
                break

            curr_obs, next_obs_dict, action = process_batch(
                batch,
                config.model.obs_encoder.num_frames,
                config.model.action_len,
                device,
            )
            height, width = get_image_size(batch, config)
            tracks = process_tracks(
                batch,
                device,
                height,
                width,
                int(config.model.get("num_points", 625)),
            )
            action_mask = batch.get("action_mask")
            if action_mask is not None:
                action_mask = action_mask.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bool(config.use_amp),
            ):
                _, info = model(
                    obs_dict=curr_obs,
                    next_obs_dict=next_obs_dict,
                    action=action,
                    tracks=tracks,
                    action_mask=action_mask,
                )
            for key, value in info.items():
                metric_sums[key] += to_scalar(value)

            target_action = unnormalize(action)
            next_obs = model.obs_encoder.encode_next_obs(next_obs_dict)
            marginal_action = model.sample_marginal_action(curr_obs)
            inverse_action = model.sample_inverse_dynamics(
                curr_obs, next_obs_dict
            )
            marginal_next_obs = model.sample_marginal_next_obs(curr_obs)
            forward_next_obs, _, _ = model.sample_forward_dynamics(
                curr_obs, action
            )
            joint_action, joint_next_obs, _, _ = model.sample_joint(curr_obs)

            metric_sums["action_mse_marginal"] += F.mse_loss(
                unnormalize(marginal_action), target_action
            ).item()
            metric_sums["action_mse_inverse"] += F.mse_loss(
                unnormalize(inverse_action), target_action
            ).item()
            metric_sums["image_mse_marginal"] += F.mse_loss(
                marginal_next_obs, next_obs
            ).item()
            metric_sums["image_mse_forward"] += F.mse_loss(
                forward_next_obs, next_obs
            ).item()
            metric_sums["image_mse_joint"] += F.mse_loss(
                joint_next_obs, next_obs
            ).item()
            metric_sums["action_mse_joint"] += F.mse_loss(
                unnormalize(joint_action), target_action
            ).item()

            if visualizations is None and is_main_process():
                visualizations = (
                    next_obs[0:1],
                    marginal_next_obs[0:1],
                    joint_next_obs[0:1],
                    forward_next_obs[0:1],
                )
            num_batches += 1

    metrics = reduce_metrics(metric_sums, num_batches, device)
    if visualizations is not None:
        labels = ("ground_truth", "marginal", "joint", "forward")
        for label, latent in zip(labels, visualizations):
            image = decode_and_plot(model, latent)
            if image is not None:
                metrics[f"images/{label}"] = image
    return metrics


def maybe_evaluate(
    config: DictConfig,
    step: int,
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    should_evaluate = (
        step % int(config.eval_every) == 0
        or step == int(config.num_steps) - 1
    )
    if not should_evaluate:
        return

    if is_main_process():
        print(f"Step {step}: running validation.")
    with timer("Validation"):
        metrics = evaluate(config, model, val_loader, device)
    if is_main_process():
        safe_log_wandb(metrics, step, prefix="eval/")
        print(f"Validation metrics: {format_metrics(metrics)}")
    distributed_barrier()


def print_model_summary(model: nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {total:,} total, {trainable:,} trainable")


def create_optimizer(
    config: DictConfig, model: nn.Module, pretrained: bool
) -> torch.optim.Optimizer:
    optimizer_args = OmegaConf.to_container(config.optimizer, resolve=True)
    if not pretrained:
        return torch.optim.AdamW(model.parameters(), **optimizer_args)

    encoder_scale = float(config.get("pretrained_encoder_lr_scale", 0.05))
    base_lr = float(optimizer_args.pop("lr"))
    encoder_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = (
            encoder_parameters
            if name.startswith("obs_encoder.img_encoder")
            else other_parameters
        )
        target.append(parameter)
    parameter_groups = [
        {"params": encoder_parameters, "lr": base_lr * encoder_scale},
        {"params": other_parameters, "lr": base_lr},
    ]
    return torch.optim.AdamW(parameter_groups, **optimizer_args)


def train(rank: int, world_size: int, config: DictConfig) -> None:
    init_distributed(rank, world_size)
    device = torch.device("cuda", rank)
    run = None
    progress = None
    try:
        set_seed(int(config.seed) * world_size + rank)
        Path(config.logdir).mkdir(parents=True, exist_ok=True)
        if is_main_process():
            run = init_wandb(config, job_type="train")
            print(OmegaConf.to_yaml(config))

        train_set = None
        if world_size == 1 or is_main_process():
            train_set = instantiate(config.dataset)
        distributed_barrier()
        if train_set is None:
            train_set = instantiate(config.dataset)
        distributed_barrier()
        val_set = (
            train_set.get_validation_dataset()
            if float(config.dataset.val_ratio) > 0
            else None
        )
        train_loader, val_loader = make_distributed_data_loader(
            train_set,
            val_set,
            int(config.batch_size),
            rank,
            world_size,
            num_workers=int(config.get("num_workers", 8)),
        )
        if len(train_loader) == 0:
            raise RuntimeError("The training loader produced no batches.")
        if is_main_process():
            print(f"Training examples: {len(train_set)}")
            if val_set is not None:
                print(f"Validation examples: {len(val_set)}")

        model = instantiate(config.model).to(device)
        pretrained_path = config.get("pretrain_checkpoint_path")
        if pretrained_path:
            model, load_info = load_from_checkpoint(model, pretrained_path)
            if is_main_process():
                ratio = load_info.get("loading_ratio")
                if ratio is not None:
                    print(f"Loaded {float(ratio):.1%} of pretrained parameters.")
        if is_main_process():
            print_model_summary(model)

        optimizer = create_optimizer(config, model, bool(pretrained_path))
        scheduler = get_scheduler(optimizer=optimizer, **config.scheduler)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.use_amp))
        step = maybe_resume_checkpoint(
            config, model, optimizer, scheduler, scaler
        )
        epoch = step // len(train_loader)

        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            static_graph=bool(config.get("static_graph", True)),
            find_unused_parameters=bool(
                config.get("find_unused_parameters", False)
            ),
        )
        progress = tqdm(
            total=int(config.num_steps),
            initial=step,
            desc="JOPAT training",
            disable=not is_main_process(),
        )
        train_stats = defaultdict(list)
        log_every = int(config.get("log_every", 10))

        while step < int(config.num_steps):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            dataset_sampler = getattr(train_set, "sampler", None)
            if hasattr(dataset_sampler, "reshuffle"):
                dataset_sampler.reshuffle()

            for batch in train_loader:
                _, info = train_one_step(
                    config,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    batch,
                    device,
                    step,
                )
                for key, value in info.items():
                    train_stats[key].append(value)

                if is_main_process() and step % log_every == 0:
                    averages = {
                        key: float(np.mean(values[-log_every:]))
                        for key, values in train_stats.items()
                    }
                    progress.set_description(
                        f"JOPAT step={step}, loss={averages['loss']:.4f}"
                    )
                    safe_log_wandb(averages, step, prefix="train/")

                if val_loader is not None:
                    maybe_evaluate(config, step, model, val_loader, device)
                maybe_collect_rollout(config, step, model, device)
                maybe_save_checkpoint(
                    config, step, model, optimizer, scheduler, scaler
                )

                step += 1
                progress.update(1)
                if step >= int(config.num_steps):
                    break
            epoch += 1

        if val_loader is not None and bool(
            config.get("run_final_evaluation", False)
        ):
            metrics = evaluate(config, model, val_loader, device)
            if is_main_process():
                safe_log_wandb(metrics, step, prefix="final/")
                print(f"Final metrics: {format_metrics(metrics)}")
    finally:
        if progress is not None:
            progress.close()
        if run is not None:
            run.finish()
        if dist.is_initialized():
            dist.destroy_process_group()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    if int(config.num_steps) <= 0:
        raise ValueError("num_steps must be positive.")
    if int(config.batch_size) <= 0:
        raise ValueError("batch_size must be positive.")
    for interval_name in ("save_every", "eval_every", "rollout_every"):
        if int(config[interval_name]) <= 0:
            raise ValueError(f"{interval_name} must be positive.")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("JOPAT training requires at least one CUDA device.")

    print("JOPAT Training")
    print(f"GPUs: {world_size}")
    print(f"Steps: {int(config.num_steps)}")
    print(f"Batch size per GPU: {int(config.batch_size)}")
    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )
    print("JOPAT training completed.")


if __name__ == "__main__":
    main()
