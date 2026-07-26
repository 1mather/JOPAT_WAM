"""Persistent trajectory storage with batched point-track preprocessing."""

import math
import os
import warnings
from multiprocessing import Lock
from os.path import expanduser, expandvars
from typing import Any, Optional, Union

import numcodecs
import numpy as np
import torch
import zarr
from tqdm import tqdm


def get_optimal_chunks(
    shape: tuple[int, ...],
    dtype: Union[str, np.dtype],
    target_chunk_bytes: float = 2e6,
) -> tuple[int, ...]:
    """Choose a chunk shape near the requested byte size."""
    itemsize = np.dtype(dtype).itemsize
    reversed_shape = list(shape[::-1])
    split_index = len(shape) - 1

    for index in range(len(shape) - 1):
        current_bytes = itemsize * np.prod(reversed_shape[:index])
        next_bytes = itemsize * np.prod(reversed_shape[: index + 1])
        if current_bytes <= target_chunk_bytes < next_bytes:
            split_index = index
            break

    reversed_chunks = reversed_shape[:split_index]
    item_chunk_bytes = itemsize * np.prod(reversed_chunks)
    next_chunk_length = min(
        reversed_shape[split_index],
        math.ceil(target_chunk_bytes / item_chunk_bytes),
    )
    reversed_chunks.append(next_chunk_length)
    reversed_chunks.extend([1] * (len(shape) - len(reversed_chunks)))
    return tuple(reversed_chunks[::-1])


class BatchProcessedTrajectoryBuffer:
    """Store episodes once and attach tracks to fixed-length window indices."""

    def __init__(
        self,
        storage_path: str,
        metadata: dict[str, dict[str, Any]],
        sequence_length: int,
        stride: int = 1,
        initial_capacity: Optional[int] = None,
        lock: Optional[Lock] = None,
        large_dataset: bool = True,
    ):
        if not storage_path:
            raise ValueError("storage_path must point to a writable Zarr directory.")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        storage_path = expandvars(expanduser(storage_path))
        self.restored = os.path.exists(storage_path)
        self.storage = zarr.DirectoryStore(storage_path)
        self.sequence_length = sequence_length
        self.stride = stride
        self.lock = Lock() if lock is None else lock
        self.root = zarr.group(store=self.storage)
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        if self.restored:
            if "episode_ends" not in self.meta or "window_indices" not in self.meta:
                raise ValueError(f"Invalid trajectory buffer at {storage_path}.")
            if any(key not in self.data for key in metadata):
                raise ValueError(
                    "Stored trajectory data does not match the requested metadata."
                )
            for key, value in metadata.items():
                if self.data[key].shape[1:] != tuple(value["shape"]):
                    raise ValueError(
                        f"Stored shape for {key!r} does not match its metadata."
                    )
            stored_sequence_length = self.meta.attrs.get("sequence_length")
            stored_stride = self.meta.attrs.get("stride")
            if (
                stored_sequence_length is not None
                and stored_sequence_length != sequence_length
            ):
                raise ValueError(
                    "Stored sequence_length does not match the requested value."
                )
            if stored_stride is not None and stored_stride != stride:
                raise ValueError(
                    "Stored stride does not match the requested value."
                )
            lengths = {
                self.data[key].shape[0] for key in self.data if key in metadata
            }
            if len(lengths) != 1:
                raise ValueError("Stored trajectory arrays have inconsistent lengths.")
            self.capacity = lengths.pop()
            print(f"Restored trajectory buffer from {storage_path}.")
            return

        initial_capacity = 10000 if initial_capacity is None else initial_capacity
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive.")

        with self.lock:
            self.capacity = initial_capacity
            self.meta.attrs["sequence_length"] = sequence_length
            self.meta.attrs["stride"] = stride
            self.meta.zeros(
                name="episode_ends",
                shape=(0,),
                dtype=np.int64,
                chunks=(max(1, min(10000, initial_capacity // 10)),),
                compressor=None,
            )
            self.meta.zeros(
                name="window_indices",
                shape=(0, 4),
                dtype=np.int64,
                chunks=(max(1, min(10000, initial_capacity)), 4),
                compressor=None,
            )

            for key, value in metadata.items():
                if "track" in key.lower():
                    continue
                shape = (initial_capacity,) + tuple(value["shape"])
                dtype = value["dtype"]
                if large_dataset:
                    is_image = dtype == np.uint8
                    chunks = self.create_chunks(shape, dtype, is_image)
                    compressor = numcodecs.Blosc(
                        cname="lz4",
                        clevel=5 if is_image else 0,
                        shuffle=numcodecs.Blosc.NOSHUFFLE,
                    )
                elif dtype == np.uint8:
                    chunks = (1,) + shape[1:]
                    compressor = numcodecs.Blosc(
                        cname="lz4",
                        clevel=5,
                        shuffle=numcodecs.Blosc.NOSHUFFLE,
                    )
                else:
                    chunks = get_optimal_chunks(shape, dtype)
                    compressor = numcodecs.Blosc(
                        cname="lz4",
                        clevel=0,
                        shuffle=numcodecs.Blosc.NOSHUFFLE,
                    )
                self.data.zeros(
                    name=key,
                    shape=shape,
                    chunks=chunks,
                    dtype=dtype,
                    compressor=compressor,
                    object_codec=numcodecs.Pickle(),
                )
        print(f"Created trajectory buffer at {storage_path}.")

    @staticmethod
    def create_chunks(shape, dtype, is_image=False):
        """Choose chunks for image or low-dimensional arrays."""
        if is_image or dtype == np.uint8:
            frame_bytes = np.prod(shape[1:]) * np.dtype(dtype).itemsize
            target_bytes = 20 * 1024 * 1024
            frames_per_chunk = max(1, int(target_bytes / frame_bytes))
            return (frames_per_chunk,) + shape[1:]
        return get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6)

    @property
    def episode_ends(self):
        return self.meta["episode_ends"]

    @property
    def window_indices(self):
        return self.meta["window_indices"]

    @property
    def num_episodes(self):
        return len(self.episode_ends)

    @property
    def num_steps(self):
        return self.episode_ends[-1] if len(self.episode_ends) > 0 else 0

    @property
    def num_windows(self):
        return len(self.window_indices)

    def _resize_if_needed(self, required_capacity: int):
        if required_capacity <= self.capacity:
            return

        new_capacity = max(required_capacity, int(self.capacity * 1.2))
        for key in self.data.keys():
            if "track" in key.lower():
                continue
            array = self.data[key]
            array.resize((new_capacity,) + array.shape[1:])
        self.capacity = new_capacity

    def add_episode(self, data: dict[str, np.ndarray]) -> int:
        """Append one episode and return the number of generated windows."""
        with self.lock:
            episode_data = {
                key: value
                for key, value in data.items()
                if "track" not in key.lower()
            }
            if not episode_data:
                raise ValueError("Episode data must contain at least one non-track array.")

            episode_lengths = np.array(
                [value.shape[0] for value in episode_data.values()]
            )
            if not np.all(episode_lengths == episode_lengths[0]):
                raise ValueError("All episode arrays must have the same leading length.")
            episode_length = int(episode_lengths[0])
            if episode_length < self.sequence_length:
                return 0

            start_index = int(self.num_steps)
            end_index = start_index + episode_length
            self._resize_if_needed(end_index)

            for key, value in episode_data.items():
                if key not in self.data:
                    raise KeyError(f"Key {key!r} is not declared in buffer metadata.")
                self.data[key][start_index:end_index] = value

            episode_id = self.num_episodes
            self.episode_ends.resize((len(self.episode_ends) + 1,))
            self.episode_ends[-1] = end_index

            max_start = episode_length - self.sequence_length
            num_windows = (max_start // self.stride) + 1
            new_indices = np.array(
                [
                    [
                        start_index,
                        local_start,
                        self.sequence_length,
                        episode_id,
                    ]
                    for local_start in range(0, max_start + 1, self.stride)
                ],
                dtype=np.int64,
            )
            old_length = len(self.window_indices)
            self.window_indices.resize((old_length + len(new_indices), 4))
            self.window_indices[old_length:] = new_indices
            return num_windows

    def _find_video_key(self, video_key: Optional[str]) -> str:
        if video_key is not None:
            if video_key not in self.data:
                raise KeyError(f"Video key {video_key!r} is not present in the buffer.")
            return video_key

        candidates = [
            key
            for key in self.data.keys()
            if "image" in key.lower() and "track" not in key.lower()
        ]
        if not candidates:
            candidates = [
                key
                for key in self.data.keys()
                if self.data[key].dtype == np.uint8
                and "track" not in key.lower()
            ]
        if not candidates:
            raise ValueError(
                f"Could not infer a video key from: {list(self.data.keys())}"
            )
        return candidates[0]

    def collect_all_windows(self, video_key: Optional[str] = None) -> list[dict]:
        """Materialize every indexed video window."""
        video_key = self._find_video_key(video_key)
        windows = []
        for window_index in range(self.num_windows):
            episode_start, local_start, length, episode_id = self.window_indices[
                window_index
            ]
            global_start = int(episode_start + local_start)
            global_end = global_start + int(length)
            windows.append(
                {
                    "window_idx": window_index,
                    "episode_id": int(episode_id),
                    "global_start": global_start,
                    "length": int(length),
                    "video": np.array(
                        self.data[video_key][global_start:global_end]
                    ),
                }
            )
        return windows

    def _ensure_track_storage(
        self,
        track_shape: tuple[int, ...],
        track_visibility_shape: Optional[tuple[int, ...]],
    ):
        if len(track_shape) != 2 or track_shape[-1] != 2:
            raise ValueError("track_shape must be (num_points, 2).")
        num_points = int(track_shape[0])
        if track_visibility_shape is not None and tuple(track_visibility_shape) != (
            num_points,
        ):
            raise ValueError(
                "track_visibility_shape must match the number of track points."
            )

        track_storage_shape = (
            self.num_windows,
            self.sequence_length,
            num_points,
            2,
        )
        visibility_storage_shape = (
            self.num_windows,
            self.sequence_length,
            num_points,
        )

        with self.lock:
            if "track" not in self.data:
                self.data.zeros(
                    name="track",
                    shape=track_storage_shape,
                    chunks=self.create_chunks(
                        track_storage_shape,
                        dtype=np.float32,
                        is_image=True,
                    ),
                    dtype=np.float32,
                    compressor=numcodecs.Blosc(cname="lz4", clevel=5),
                )
            if "track_visibility" not in self.data:
                self.data.zeros(
                    name="track_visibility",
                    shape=visibility_storage_shape,
                    chunks=self.create_chunks(
                        visibility_storage_shape,
                        dtype=np.float32,
                        is_image=False,
                    ),
                    dtype=np.float32,
                    compressor=numcodecs.Blosc(cname="lz4", clevel=5),
                )

        if self.data["track"].shape != track_storage_shape:
            raise ValueError(
                "Stored track shape does not match the current window metadata."
            )
        if self.data["track_visibility"].shape != visibility_storage_shape:
            raise ValueError(
                "Stored visibility shape does not match the current window metadata."
            )

    def _ensure_completion_mask(self):
        if "track_done" not in self.meta:
            with self.lock:
                if "track_done" not in self.meta:
                    self.meta.zeros(
                        name="track_done",
                        shape=(self.num_windows,),
                        dtype=np.bool_,
                        chunks=(min(100000, max(1, self.num_windows)),),
                        compressor=None,
                    )
                    self._infer_legacy_completion_prefix()
        elif self.meta["track_done"].shape[0] != self.num_windows:
            raise ValueError(
                "The track completion mask does not match the number of windows. "
                "Rebuild the window buffer."
            )

    def _infer_legacy_completion_prefix(self):
        if "track_visibility" not in self.data or self.num_windows == 0:
            return

        def is_done(window_index: int) -> bool:
            visibility = np.array(self.data["track_visibility"][window_index])
            return bool(np.any(visibility != 0))

        try:
            if is_done(self.num_windows - 1):
                self.meta["track_done"][:] = True
                return
            if not is_done(0):
                return
            low, high = 0, self.num_windows - 1
            while low < high:
                middle = (low + high + 1) // 2
                if is_done(middle):
                    low = middle
                else:
                    high = middle - 1
            if is_done(low):
                self.meta["track_done"][: low + 1] = True
        except (IndexError, TypeError, ValueError):
            return

    def batch_process_tracks(
        self,
        cotracker_model,
        batch_size: int = 64,
        grid_size: int = 25,
        device: str = "cuda",
        track_shape: Optional[tuple[int, ...]] = None,
        track_visibility_shape: Optional[tuple[int, ...]] = None,
        video_key: Optional[str] = None,
    ):
        """Run CoTracker in batches and persist tracks for every window."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if track_shape is None:
            raise ValueError("track_shape is required.")

        video_key = self._find_video_key(video_key)
        self._ensure_track_storage(track_shape, track_visibility_shape)
        self._ensure_completion_mask()

        track_done = np.array(self.meta["track_done"][:], dtype=bool)
        pending_indices = np.nonzero(~track_done)[0]
        if pending_indices.size == 0:
            print("Tracks are already available for every window.")
            return

        sequence_length = self.sequence_length
        num_points = int(track_shape[0])
        zero_track = np.zeros(
            (sequence_length, num_points, 2),
            dtype=np.float32,
        )
        zero_visibility = np.zeros(
            (sequence_length, num_points),
            dtype=np.float32,
        )

        def write_zeros(window_index: int):
            self.data["track"][window_index] = zero_track
            self.data["track_visibility"][window_index] = zero_visibility
            self.meta["track_done"][window_index] = True

        progress = tqdm(
            total=int(pending_indices.size),
            desc="Precomputing point tracks",
        )
        for batch_start in range(0, int(pending_indices.size), batch_size):
            batch_windows = []
            batch_indices = []
            selected_indices = pending_indices[
                batch_start : batch_start + batch_size
            ]

            for raw_window_index in selected_indices:
                window_index = int(raw_window_index)
                episode_start, local_start, length, episode_id = (
                    self.window_indices[window_index]
                )
                global_start = int(episode_start + local_start)
                global_end = global_start + int(length)
                video = np.array(self.data[video_key][global_start:global_end])
                if video.shape[0] != sequence_length:
                    warnings.warn(
                        "Skipping a window with an unexpected temporal length "
                        f"(window={window_index}, episode={int(episode_id)}).",
                        RuntimeWarning,
                    )
                    write_zeros(window_index)
                    progress.update(1)
                    continue
                batch_windows.append(video)
                batch_indices.append(window_index)

            if not batch_windows:
                continue

            batch_array = np.stack(batch_windows)
            batch_tensor = (
                torch.from_numpy(batch_array)
                .float()
                .permute(0, 1, 4, 2, 3)
                .contiguous()
                .to(device)
            )

            try:
                with torch.no_grad():
                    predicted_tracks, predicted_visibility = cotracker_model(
                        batch_tensor,
                        grid_size=grid_size,
                        grid_query_frame=0,
                    )
                predicted_tracks = predicted_tracks.cpu().numpy()
                predicted_visibility = predicted_visibility.cpu().numpy()
                for result_index, window_index in enumerate(batch_indices):
                    try:
                        self.data["track"][window_index] = predicted_tracks[
                            result_index
                        ]
                        self.data["track_visibility"][window_index] = (
                            predicted_visibility[result_index]
                        )
                        self.meta["track_done"][window_index] = True
                    except (IndexError, TypeError, ValueError) as error:
                        warnings.warn(
                            f"Could not store tracks for window {window_index}; "
                            f"writing zeros instead ({error}).",
                            RuntimeWarning,
                        )
                        write_zeros(window_index)
                progress.update(len(batch_indices))
                continue
            except RuntimeError as error:
                warnings.warn(
                    "Batched CoTracker inference failed; retrying each window "
                    f"individually ({error}).",
                    RuntimeWarning,
                )

            for video, window_index in zip(batch_windows, batch_indices):
                try:
                    tensor = (
                        torch.from_numpy(video[None])
                        .float()
                        .permute(0, 1, 4, 2, 3)
                        .contiguous()
                        .to(device)
                    )
                    with torch.no_grad():
                        tracks, visibility = cotracker_model(
                            tensor,
                            grid_size=grid_size,
                            grid_query_frame=0,
                        )
                    self.data["track"][window_index] = tracks[0].cpu().numpy()
                    self.data["track_visibility"][window_index] = (
                        visibility[0].cpu().numpy()
                    )
                    self.meta["track_done"][window_index] = True
                except (IndexError, RuntimeError, TypeError, ValueError) as error:
                    warnings.warn(
                        f"CoTracker failed for window {window_index}; "
                        f"writing zeros instead ({error}).",
                        RuntimeWarning,
                    )
                    write_zeros(window_index)
                progress.update(1)
        progress.close()

    def get_window(self, window_idx: int) -> dict[str, np.ndarray]:
        """Return one window, including any precomputed track arrays."""
        if window_idx < 0 or window_idx >= self.num_windows:
            raise IndexError("Window index is out of range.")

        episode_start, local_start, length, _ = self.window_indices[window_idx]
        global_start = int(episode_start + local_start)
        global_end = global_start + int(length)
        result = {}
        for key in self.data.keys():
            if "track" in key.lower():
                result[key] = np.array(self.data[key][window_idx])
            else:
                result[key] = np.array(
                    self.data[key][global_start:global_end]
                )
        return result

    def trim(self):
        """Trim unused episode capacity without changing window arrays."""
        with self.lock:
            used_steps = int(self.num_steps)
            if used_steps >= self.capacity:
                return
            for key in self.data.keys():
                if "track" in key.lower():
                    continue
                array = self.data[key]
                array.resize((used_steps,) + array.shape[1:])
            self.capacity = used_steps

    def __repr__(self):
        return (
            "BatchProcessedTrajectoryBuffer(\n"
            f"  episodes={self.num_episodes},\n"
            f"  total_steps={self.num_steps},\n"
            f"  windows={self.num_windows},\n"
            f"  sequence_length={self.sequence_length},\n"
            f"  stride={self.stride},\n"
            f"  capacity={self.capacity}\n"
            ")"
        )

    def __len__(self):
        return self.num_windows

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
