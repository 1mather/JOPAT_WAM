import math
import os
from multiprocessing import Lock
from os.path import expanduser, expandvars
from typing import Any, Optional, Union

import numcodecs
import numpy as np
import zarr


def get_optimal_chunks(
    shape: tuple[int, ...], dtype: Union[str, np.dtype], target_chunk_bytes: float = 2e6
) -> tuple[int, ...]:
    """Choose a chunk shape near the requested byte size."""
    itemsize = np.dtype(dtype).itemsize
    rshape = list(shape[::-1])

    split_idx = len(shape) - 1
    for i in range(len(shape) - 1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[: i + 1])
        if this_chunk_bytes <= target_chunk_bytes < next_chunk_bytes:
            split_idx = i
            break

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rchunks)
    next_chunk_length = min(
        rshape[split_idx], math.ceil(target_chunk_bytes / item_chunk_bytes)
    )
    rchunks.append(next_chunk_length)

    rchunks.extend([1] * (len(shape) - len(rchunks)))
    chunks = tuple(rchunks[::-1])
    return chunks


class CompressedTrajectoryBuffer:
    """Store variable-length episodes in compressed Zarr arrays."""

    def __init__(
        self,
        storage_path: str,
        metadata: dict[str, dict[str, Any]],
        capacity: Optional[int] = None,
        lock: Optional[Lock] = None,
        large_dataset: bool = True,
    ):
        """Restore an existing buffer or create one with the given capacity."""
        if not storage_path:
            raise ValueError("storage_path must point to a writable Zarr directory.")
        storage_path = expandvars(expanduser(storage_path))
        self.restored = os.path.exists(storage_path)
        self.storage = zarr.DirectoryStore(storage_path)

        self.lock = Lock() if lock is None else lock

        self.root = zarr.group(store=self.storage)
        self.data = self.root.require_group("data")
        self.meta = self.root.require_group("meta")

        if self.restored:
            print(f"Restoring buffer from {storage_path}")
            assert "episode_ends" in self.meta
            assert all(key in self.data for key in metadata)
            assert all(
                self.data[key].shape[1:] == value["shape"]
                for key, value in metadata.items()
            )

            lengths = {self.data[key].shape[0] for key in self.data}
            assert len(lengths) == 1, "Inconsistent data lengths in the buffer"
            self.capacity = lengths.pop()
        else:
            with self.lock:
                print(f"Creating new buffer at {storage_path}")
                assert capacity is not None, "Capacity must be specified for new buffer"
                self.capacity = capacity

                self.meta.zeros(
                    name="episode_ends",
                    shape=(0,),
                    dtype=np.int64,
                    compressor=None,
                )

                for key, value in metadata.items():
                    shape = (capacity,) + tuple(value["shape"])
                    dtype = value["dtype"]
                    if not large_dataset:
                        if dtype == np.uint8:
                            chunks = (1,) + shape[1:]
                            compressor = numcodecs.Blosc(
                                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
                            )
                        else:
                            chunks = get_optimal_chunks(shape, dtype)
                            compressor = numcodecs.Blosc(
                                cname="lz4", clevel=0, shuffle=numcodecs.Blosc.NOSHUFFLE
                            )
                    else:
                        if dtype == np.uint8:
                            is_image = True
                            compressor = numcodecs.Blosc(
                                cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
                            )
                        else:
                            is_image = False
                            compressor = numcodecs.Blosc(
                                cname="lz4", clevel=0, shuffle=numcodecs.Blosc.NOSHUFFLE
                            )
                        chunks = self.create_chunks(shape, dtype, is_image)
                    self.data.zeros(
                        name=key,
                        shape=shape,
                        chunks=chunks,
                        dtype=dtype,
                        compressor=compressor,
                        object_codec=numcodecs.Pickle(),
                    )

    @staticmethod
    def create_chunks(shape, dtype, is_image=False):
        """Choose chunks for image or low-dimensional arrays."""
        if is_image or dtype == np.uint8:
            frame_bytes = np.prod(shape[1:]) * np.dtype(dtype).itemsize
            target_bytes = 20 * 1024 * 1024
            frames_per_chunk = max(1, int(target_bytes / frame_bytes))
            chunks = (frames_per_chunk,) + shape[1:]
        else:
            chunks = get_optimal_chunks(shape, dtype, target_chunk_bytes=2e6)
        return chunks

    @property
    def episode_ends(self) -> np.ndarray:
        return self.meta["episode_ends"]

    @property
    def num_episodes(self) -> int:
        return len(self.episode_ends)

    @property
    def num_steps(self) -> int:
        return self.episode_ends[-1] if len(self.episode_ends) > 0 else 0

    def add_episode(self, data: dict[str, np.ndarray]):
        with self.lock:
            episode_lens = np.array([v.shape[0] for v in data.values()])
            assert np.all(episode_lens == episode_lens[0])
            episode_len = episode_lens[0]

            start_ind = self.num_steps
            end_ind = start_ind + episode_len
            if end_ind > self.capacity:
                raise RuntimeError("Buffer capacity exceeded")

            for key, value in data.items():
                arr = self.data[key]
                arr[start_ind:end_ind] = value

            self.episode_ends.resize(len(self.episode_ends) + 1)
            self.episode_ends[-1] = end_ind

            if self.episode_ends.chunks[0] < self.episode_ends.shape[0]:
                new_chunk_len = math.ceil(self.episode_ends.shape[0] * 1.5)
                new_chunks = (new_chunk_len,) + self.episode_ends.chunks[1:]
                self.meta.move("episode_ends", "_temp")
                zarr.copy(
                    source=self.meta["_temp"],
                    dest=self.meta,
                    name="episode_ends",
                    chunks=new_chunks,
                    compressor=None,
                )
                del self.meta["_temp"]

    def __repr__(self) -> str:
        return str(self.root.tree())

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data
