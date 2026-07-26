"""Window samplers for trajectory buffers with precomputed tracks."""

from typing import Optional

import numpy as np


class WindowSampler:
    """Sample indexed windows from a trajectory buffer."""

    def __init__(
        self,
        buffer,
        train_mask: Optional[np.ndarray] = None,
        shuffle: bool = True,
    ):
        self.buffer = buffer
        self.shuffle = shuffle
        total_windows = self.buffer.num_windows

        if train_mask is None:
            self.indices = np.arange(total_windows)
        else:
            if len(train_mask) != total_windows:
                raise ValueError(
                    "train_mask length must match the number of windows."
                )
            self.indices = np.where(train_mask)[0]
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx: int) -> dict:
        """Return the window selected by a sampler-local index."""
        return self.buffer.get_window(int(self.indices[idx]))

    def reshuffle(self):
        """Shuffle window order in place."""
        if self.shuffle:
            np.random.shuffle(self.indices)


class EpisodeBasedWindowSampler(WindowSampler):
    """Select windows by episode ID to prevent split leakage."""

    def __init__(
        self,
        buffer,
        train_episode_ids: Optional[np.ndarray] = None,
        shuffle: bool = True,
    ):
        self.buffer = buffer
        self.shuffle = shuffle
        window_indices = buffer.window_indices[:]
        episode_ids = window_indices[:, 3]

        if train_episode_ids is None:
            self.indices = np.arange(len(episode_ids))
        else:
            self.indices = np.where(
                np.isin(episode_ids, train_episode_ids)
            )[0]
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_train_val_samplers(
    buffer,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    """Create episode-disjoint training and validation samplers."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    num_episodes = buffer.num_episodes
    if num_episodes < 2:
        raise ValueError(
            "At least two episodes are required for a train/validation split."
        )

    all_episode_ids = np.arange(num_episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_episode_ids)

    num_validation = round(num_episodes * val_ratio)
    num_validation = min(max(num_validation, 1), num_episodes - 1)
    validation_episode_ids = all_episode_ids[:num_validation]
    training_episode_ids = all_episode_ids[num_validation:]

    training_sampler = EpisodeBasedWindowSampler(
        buffer,
        train_episode_ids=training_episode_ids,
        shuffle=shuffle,
    )
    validation_sampler = EpisodeBasedWindowSampler(
        buffer,
        train_episode_ids=validation_episode_ids,
        shuffle=False,
    )
    return training_sampler, validation_sampler
