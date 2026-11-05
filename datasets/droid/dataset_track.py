"""DROID dataset with per-window CoTracker preprocessing."""

import copy
import pathlib
import warnings
from typing import Optional

import dask.array as da
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


from ..utils.batch_processed_buffer import BatchProcessedTrajectoryBuffer
from ..utils.buffer import CompressedTrajectoryBuffer
from ..utils.normalizer import LinearNormalizer, NestedDictLinearNormalizer
from ..utils.obs_utils import unflatten_obs
from ..utils.window_sampler import WindowSampler, create_train_val_samplers

try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError as error:
    CoTrackerPredictor = None
    _COTRACKER_IMPORT_ERROR = error


class DroidTrackDataset(Dataset):
    """Load DROID trajectories and precompute tracks for each window."""

    def __init__(
        self,
        name: str,
        buffer_path: str,
        subepisode_buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        val_ratio: float = 0.0,
        history_len: int = 1,
        normalize_lowdim: bool = False,
        normalize_action: bool = False,
        num_workers: int = 24,
        cotracker_checkpoint: Optional[str] = None,
        device: str = "cuda",
        offline: bool = True,
        window_len: int = 21,
        track_len: int = 19,
        reference_frame: int = 1,
        visibility_threshold: float = 0.9,
        grid_size: int = 25,
        cotracker_batch_size: int = 64,
        stride: int = 1,
        obs_num_frames: int = 2,
        action_free: bool = False,
    ):
        self.name = name
        self.seq_len = seq_len
        self.history_len = history_len
        self.num_workers = num_workers

        self.cotracker_checkpoint = cotracker_checkpoint
        self.device = device
        self.offline = offline
        self.window_len = window_len
        self.track_len = track_len
        self.reference_frame = reference_frame
        self.visibility_threshold = visibility_threshold
        self.grid_size = grid_size
        self.action_free = action_free
        self.cotracker_batch_size = cotracker_batch_size
        self.stride = stride
        self.subepisode_buffer_path = subepisode_buffer_path

        obs_shape_meta = shape_meta["obs"]
        self._image_shapes: dict[str, tuple[int, ...]] = {}
        self._lowdim_shapes: dict[str, tuple[int, ...]] = {}
        for key, attr in obs_shape_meta.items():
            obs_type = attr["type"]
            obs_shape = tuple(attr["shape"])
            if obs_type == "rgb":
                self._image_shapes[key] = obs_shape
            elif obs_type == "low_dim":
                self._lowdim_shapes[key] = obs_shape
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        self._action_shape = tuple(shape_meta["action"]["shape"])

        self._track_shape = (
            tuple(shape_meta["track"]["shape"]) if "track" in shape_meta else None
        )
        self._track_visibility_shape = (
            tuple(shape_meta["track_visibility"]["shape"])
            if "track_visibility" in shape_meta
            else None
        )

        self.buffer_dir = pathlib.Path(buffer_path).parent
        self.episode_buffer = self._init_episode_buffer(buffer_path)
        self.subepisode_buffer = self._init_window_buffer()

        if val_ratio > 0:
            self.train_sampler, self.val_sampler = create_train_val_samplers(
                self.subepisode_buffer,
                val_ratio=val_ratio,
                shuffle=True,
                seed=42,
            )
            self.sampler = self.train_sampler
        else:
            self.sampler = WindowSampler(
                self.subepisode_buffer,
                train_mask=None,
                shuffle=True,
            )
            self.train_sampler = self.sampler
            self.val_sampler = None

        if normalize_lowdim:
            self.lowdim_normalizer = self._init_lowdim_normalizer()

        if normalize_action:
            self.action_normalizer = self._init_action_normalizer()

    def _init_episode_buffer(self, buffer_path: str) -> CompressedTrajectoryBuffer:
        if not pathlib.Path(buffer_path).expanduser().exists():
            raise FileNotFoundError(f"Episode buffer not found at {buffer_path}.")
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        buffer = CompressedTrajectoryBuffer(storage_path=buffer_path, metadata=metadata)
        return buffer

    def _init_lowdim_normalizer(self) -> NestedDictLinearNormalizer:
        normalizer_stats_path = self.buffer_dir / "lowdim_normalizer_stats.npz"
        if normalizer_stats_path.exists():
            print(f"Loading lowdim normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return NestedDictLinearNormalizer(stats)

        stats = {}
        for key in self._lowdim_shapes.keys():
            data = da.from_zarr(self.subepisode_buffer[f"obs.{key}"])
            min_val = data.min(axis=0).compute()
            max_val = data.max(axis=0).compute()
            scale = (max_val - min_val) / 2.0
            offset = (max_val + min_val) / 2.0
            stats[key] = (scale, offset)

        np.savez(normalizer_stats_path, **stats)
        return NestedDictLinearNormalizer(stats)

    def _init_action_normalizer(self) -> LinearNormalizer:
        normalizer_stats_name = f"action_normalizer_stats_len{self.seq_len}.npz"
        normalizer_stats_path = self.buffer_dir / normalizer_stats_name
        if normalizer_stats_path.exists():
            print(f"Loading action normalizer stats from {normalizer_stats_path}")
            stats = np.load(normalizer_stats_path)
            return LinearNormalizer(stats["scale"], stats["offset"])

        actions = da.from_zarr(self.subepisode_buffer["action"])
        min_action = actions.min(axis=0).compute()
        max_action = actions.max(axis=0).compute()
        scale = (max_action - min_action) / 2.0
        offset = (max_action + min_action) / 2.0

        np.savez(normalizer_stats_path, scale=scale, offset=offset)
        return LinearNormalizer(scale, offset)

    def load_cotracker_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        offline: bool = True,
        window_len: int = 60,
    ):
        if CoTrackerPredictor is None:
            raise ImportError(
                "CoTracker is required to precompute point tracks. Install a "
                "compatible CoTracker package that provides "
                "`cotracker.predictor.CoTrackerPredictor`."
            ) from _COTRACKER_IMPORT_ERROR

        if offline:
            window_len = 60
        else:
            window_len = 16

        if checkpoint_path and "online" in checkpoint_path.lower() and offline:
            warnings.warn(
                "The checkpoint appears to use the online predictor; switching "
                "to online mode.",
                RuntimeWarning,
            )
            window_len = 16
            offline = False

        model = CoTrackerPredictor(
            checkpoint=checkpoint_path,
            offline=offline,
            window_len=window_len,
        )
        model = model.to(device)
        model.eval()
        return model

    def _init_window_buffer(self) -> BatchProcessedTrajectoryBuffer:
        metadata = {}
        for key, shape in self._image_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.uint8}
        for key, shape in self._lowdim_shapes.items():
            metadata[f"obs.{key}"] = {"shape": shape, "dtype": np.float32}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        buffer = BatchProcessedTrajectoryBuffer(
            storage_path=self.subepisode_buffer_path,
            metadata=metadata,
            sequence_length=self.seq_len,
            stride=self.stride,
            initial_capacity=max(10000, int(self.episode_buffer.num_steps)),
        )

        if not buffer.restored:
            for episode_idx in tqdm(
                range(self.episode_buffer.num_episodes),
                desc="Building DROID trajectory windows",
                total=self.episode_buffer.num_episodes,
                dynamic_ncols=True,
            ):
                episode_start = (
                    0
                    if episode_idx == 0
                    else int(
                        self.episode_buffer.episode_ends[episode_idx - 1]
                    )
                )
                episode_end = int(
                    self.episode_buffer.episode_ends[episode_idx]
                )
                episode_data = {}
                for key in self._image_shapes:
                    episode_data[f"obs.{key}"] = np.array(
                        self.episode_buffer[f"obs.{key}"][
                            episode_start:episode_end
                        ]
                    )
                for key in self._lowdim_shapes:
                    episode_data[f"obs.{key}"] = np.array(
                        self.episode_buffer[f"obs.{key}"][
                            episode_start:episode_end
                        ]
                    )
                episode_data["action"] = np.array(
                    self.episode_buffer["action"][episode_start:episode_end]
                )
                buffer.add_episode(episode_data)

        tracks_pending = self._track_shape is not None and (
            "track" not in buffer.data
            or "track_visibility" not in buffer.data
            or "track_done" not in buffer.meta
            or not np.all(np.asarray(buffer.meta["track_done"][:], dtype=bool))
        )
        if tracks_pending:
            if self.cotracker_checkpoint is None:
                raise ValueError(
                    "shape_meta contains `track`, but `cotracker_checkpoint` is None."
                )
            if self._track_visibility_shape is None:
                raise ValueError(
                    "shape_meta contains `track`, but missing `track_visibility`."
                )

            cotracker_model = self.load_cotracker_model(
                checkpoint_path=self.cotracker_checkpoint,
                device=self.device,
                offline=self.offline,
                window_len=self.window_len,
            )

            video_key = None
            if self._image_shapes:
                video_key = f"obs.{next(iter(self._image_shapes))}"

            buffer.batch_process_tracks(
                cotracker_model=cotracker_model,
                batch_size=self.cotracker_batch_size,
                grid_size=self.grid_size,
                device=self.device,
                track_shape=self._track_shape,
                track_visibility_shape=self._track_visibility_shape,
                video_key=video_key,
                track_len=self.track_len,
                reference_frame=self.reference_frame,
                visibility_threshold=self.visibility_threshold,
            )

        buffer.trim()
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return (
            "<DroidTrackDataset>\n"
            f"name: {self.name}\n"
            f"num_windows: {len(self)}\n"
            f"{self.subepisode_buffer}"
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        if hasattr(self, "lowdim_normalizer"):
            for key in self._lowdim_shapes.keys():
                data[f"obs.{key}"] = self.lowdim_normalizer[key](data[f"obs.{key}"])

        if hasattr(self, "action_normalizer"):
            data["action"] = self.action_normalizer(data["action"])

        data = {k: torch.from_numpy(v) for k, v in data.items()}
        data = unflatten_obs(data)
        data["action_mask"] = torch.tensor(
            0 if self.action_free else 1, dtype=torch.bool
        )
        return data

    def get_validation_dataset(self):
        if self.val_sampler is None:
            raise ValueError("No validation set. Create dataset with val_ratio > 0.")
        val_set = copy.copy(self)
        val_set.sampler = self.val_sampler
        return val_set
