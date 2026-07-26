"""JOPAT dataset with per-window CoTracker preprocessing."""

import copy
import warnings
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from ..robomimic.dataset import RobomimicDataset
from ..utils.batch_processed_buffer import BatchProcessedTrajectoryBuffer
from ..utils.obs_utils import unflatten_obs
from ..utils.window_sampler import WindowSampler, create_train_val_samplers

try:
    from cotracker.predictor import CoTrackerPredictor
except ImportError as error:
    CoTrackerPredictor = None
    _COTRACKER_IMPORT_ERROR = error


class JOPATTrackDataset(RobomimicDataset):
    """Load robot trajectories and precompute point tracks for each window."""

    def __init__(
        self,
        name: str,
        hdf5_path_globs: str,
        buffer_path: str,
        shape_meta: dict,
        seq_len: int,
        val_ratio: float = 0.0,
        subsample_ratio: float = 1.0,
        flip_rgb: bool = False,
        rollout_name: Optional[str] = None,
        cotracker_checkpoint: Optional[str] = None,
        device: str = "cuda",
        offline: bool = True,
        window_len: int = 19,
        grid_size: int = 25,
        subepisode_buffer_path: Optional[str] = None,
        use_language: bool = False,
        cotracker_batch_size: int = 64,
        stride: int = 1,
    ):
        if "track" not in shape_meta or "track_visibility" not in shape_meta:
            raise ValueError(
                "shape_meta must define both track and track_visibility."
            )

        self.rollout_name = rollout_name
        self.cotracker_checkpoint = cotracker_checkpoint
        self.device = device
        self.offline = offline
        self.window_len = window_len
        self.grid_size = grid_size
        self.subepisode_buffer_path = subepisode_buffer_path
        self.cotracker_batch_size = cotracker_batch_size
        self.stride = stride

        super().__init__(
            name=name,
            hdf5_path_globs=hdf5_path_globs,
            buffer_path=buffer_path,
            shape_meta=shape_meta,
            seq_len=seq_len,
            val_ratio=val_ratio,
            subsample_ratio=subsample_ratio,
            flip_rgb=flip_rgb,
            use_language=use_language,
        )

        self._track_shape = tuple(shape_meta["track"]["shape"])
        self._track_visibility_shape = tuple(
            shape_meta["track_visibility"]["shape"]
        )
        self.episode_buffer = self.buffer
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

    @staticmethod
    def load_cotracker_model(
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        offline: bool = True,
        window_len: int = 60,
    ):
        """Load a CoTracker predictor for preprocessing."""
        if CoTrackerPredictor is None:
            raise ImportError(
                "CoTracker is required to precompute point tracks. Install a "
                "compatible CoTracker package that provides "
                "`cotracker.predictor.CoTrackerPredictor`."
            ) from _COTRACKER_IMPORT_ERROR

        window_len = 60 if offline else 16
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

        if self.use_language:
            metadata["language"] = {"shape": (), "dtype": "U256"}
            metadata["input_ids"] = {
                "shape": (self.max_language_length,),
                "dtype": np.int64,
            }
            metadata["attention_mask"] = {
                "shape": (self.max_language_length,),
                "dtype": np.int64,
            }

        buffer = BatchProcessedTrajectoryBuffer(
            storage_path=self.subepisode_buffer_path,
            metadata=metadata,
            sequence_length=self.seq_len,
            stride=self.stride,
            initial_capacity=10000,
        )
        if not buffer.restored:
            for episode_index in tqdm(
                range(self.episode_buffer.num_episodes),
                desc="Building trajectory windows",
            ):
                episode_start = (
                    0
                    if episode_index == 0
                    else int(
                        self.episode_buffer.episode_ends[episode_index - 1]
                    )
                )
                episode_end = int(
                    self.episode_buffer.episode_ends[episode_index]
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

                if self.use_language:
                    for key in ("language", "input_ids", "attention_mask"):
                        episode_data[key] = np.array(
                            self.episode_buffer[key][episode_start:episode_end]
                        )
                buffer.add_episode(episode_data)

        tracks_pending = (
            "track" not in buffer.data
            or "track_visibility" not in buffer.data
            or "track_done" not in buffer.meta
            or not np.all(np.asarray(buffer.meta["track_done"][:], dtype=bool))
        )
        if tracks_pending:
            cotracker_model = self.load_cotracker_model(
                checkpoint_path=self.cotracker_checkpoint,
                device=self.device,
                offline=self.offline,
                window_len=self.window_len,
            )
            video_key = (
                f"obs.{next(iter(self._image_shapes))}"
                if self._image_shapes
                else None
            )
            buffer.batch_process_tracks(
                cotracker_model=cotracker_model,
                batch_size=self.cotracker_batch_size,
                grid_size=self.grid_size,
                device=self.device,
                track_shape=self._track_shape,
                track_visibility_shape=self._track_visibility_shape,
                video_key=video_key,
            )
        buffer.trim()
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return (
            "<JOPATTrackDataset>\n"
            f"name: {self.name}\n"
            f"num_windows: {len(self)}\n"
            f"{self.subepisode_buffer}"
        )

    def __getitem__(self, idx: int) -> dict:
        data = self.sampler.sample_sequence(idx)
        converted_data = {}
        for key, value in data.items():
            if (
                isinstance(value, np.ndarray)
                and value.dtype.kind in {"U", "S", "O"}
            ):
                converted_data[key] = value
            else:
                converted_data[key] = torch.from_numpy(value)
        return unflatten_obs(converted_data)

    def get_validation_dataset(self):
        """Return a shallow dataset view backed by the validation sampler."""
        if self.val_sampler is None:
            raise ValueError(
                "No validation set is available; initialize with val_ratio > 0."
            )
        validation_set = copy.copy(self)
        validation_set.sampler = self.val_sampler
        return validation_set
