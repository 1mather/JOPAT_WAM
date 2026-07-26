import copy
import os
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import CLIPTokenizer

from ..utils.buffer import CompressedTrajectoryBuffer
from ..utils.file_utils import glob_all
from ..utils.obs_utils import unflatten_obs
from ..utils.sampler import TrajectorySampler


class RobomimicDataset(Dataset):
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
        use_language: bool = False,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_language_length: int = 32,
    ):
        self.name = name
        self.seq_len = seq_len
        self.flip_rgb = flip_rgb
        self.use_language = use_language
        self.max_language_length = max_language_length
        if self.use_language:
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None

        obs_shape_meta = shape_meta["obs"]
        self._image_shapes = {}
        self._lowdim_shapes = {}
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
        self.buffer = self._init_buffer(hdf5_path_globs, buffer_path)

        num_episodes = self.buffer.num_episodes
        val_mask = np.zeros(num_episodes, dtype=bool)
        if val_ratio > 0:
            num_val_episodes = round(val_ratio * num_episodes)
            num_val_episodes = min(max(num_val_episodes, 1), num_episodes - 1)
            rng = np.random.default_rng(seed=0)
            val_inds = rng.choice(num_episodes, num_val_episodes, replace=False)
            val_mask[val_inds] = True
        self.val_mask = val_mask
        self.train_mask = ~val_mask

        if subsample_ratio < 1.0:
            train_indices = np.where(self.train_mask)[0]
            num_train_episodes = len(train_indices)
            num_subsampled = round(num_train_episodes * subsample_ratio)
            num_subsampled = max(1, num_subsampled)

            subsampled_train_mask = np.zeros(num_episodes, dtype=bool)
            rng = np.random.default_rng(seed=1)
            sampled_indices = rng.choice(train_indices, num_subsampled, replace=False)
            subsampled_train_mask[sampled_indices] = True
            self.train_mask = subsampled_train_mask

        self.sampler = TrajectorySampler(self.buffer, self.seq_len, self.train_mask)

    def _extract_language_from_path(self, hdf5_path: str) -> str:
        """Extract a language instruction from a demonstration filename."""
        filename = os.path.basename(hdf5_path)
        filename = filename.replace(".hdf5", "")
        filename = filename.replace("_demo", "")
        scene_pattern = r"^[A-Z_]+SCENE\d+_"
        match = re.match(scene_pattern, filename)
        if match:
            language_instruction = filename[match.end() :]
        else:
            language_instruction = filename
        return language_instruction.replace("_", " ")

    def _tokenize_language(self, text: str) -> dict:
        """Tokenize an instruction and return IDs plus an attention mask."""
        if not self.use_language or self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Set use_language=True.")

        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_language_length,
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
        }

    def _init_buffer(self, hdf5_path_globs, buffer_path):
        hdf5_paths = glob_all(hdf5_path_globs)

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

        capacity = 0
        num_episodes = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    capacity += demo["actions"].shape[0]
                num_episodes += len(demos)

        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )

        if buffer.restored:
            return buffer

        pbar = tqdm(total=num_episodes, desc="Loading episodes to buffer")
        for hdf5_path in hdf5_paths:
            if self.use_language:
                language_instruction = self._extract_language_from_path(hdf5_path)
                tokenized = self._tokenize_language(language_instruction)

            with h5py.File(hdf5_path) as f:
                demos = f["data"]
                for i in range(len(demos)):
                    demo = demos[f"demo_{i}"]
                    episode = {}
                    for key in self._image_shapes.keys():
                        if self.flip_rgb:
                            episode[f"obs.{key}"] = demo["obs"][key][:][:, ::-1]
                        else:
                            episode[f"obs.{key}"] = demo["obs"][key][:]
                    for key in self._lowdim_shapes.keys():
                        episode[f"obs.{key}"] = demo["obs"][key][:]
                    episode["action"] = demo["actions"][:]

                    if self.use_language:
                        episode_len = demo["actions"].shape[0]
                        episode["language"] = np.array(
                            [language_instruction] * episode_len,
                            dtype="U256",
                        )
                        episode["input_ids"] = np.tile(
                            tokenized["input_ids"],
                            (episode_len, 1),
                        )
                        episode["attention_mask"] = np.tile(
                            tokenized["attention_mask"],
                            (episode_len, 1),
                        )

                    buffer.add_episode(episode)
                    pbar.update(1)
        pbar.close()
        return buffer

    def __len__(self) -> int:
        return len(self.sampler)

    def __repr__(self) -> str:
        return f"<RobomimicDataset>\nname: {self.name}\nnum_samples: {len(self)}\n{self.buffer}"

    def __getitem__(self, idx: int) -> dict:
        data = self.sampler.sample_sequence(idx)
        converted = {
            key: (
                value
                if isinstance(value, np.ndarray)
                and value.dtype.kind in {"U", "S", "O"}
                else torch.from_numpy(value)
            )
            for key, value in data.items()
        }
        return unflatten_obs(converted)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = self.val_mask
        val_set.sampler = TrajectorySampler(self.buffer, self.seq_len, self.val_mask)
        return val_set
