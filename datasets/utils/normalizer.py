from typing import Union

import numpy as np


class LinearNormalizer:
    """Linearly normalize data to approximately [-1, 1]."""

    def __init__(self, scale: np.ndarray, offset: np.ndarray):
        self.scale = scale
        self.offset = offset

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / self.scale

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.offset


class NestedDictLinearNormalizer(dict):
    """Apply linear normalization recursively to matching dictionary keys."""

    def __init__(self, stats: dict[str, Union[tuple[np.ndarray, np.ndarray], dict]]):
        super().__init__()
        for k, v in stats.items():
            if isinstance(v, dict):
                self[k] = NestedDictLinearNormalizer(v)
            else:
                self[k] = LinearNormalizer(np.array(v[0]), np.array(v[1]))

    def __call__(self, x: dict) -> dict:
        return {k: self[k](v) if k in self.keys() else v for k, v in x.items()}

    def reconstruct(self, x: dict) -> dict:
        return {
            k: self[k].reconstruct(v) if k in self.keys() else v for k, v in x.items()
        }
