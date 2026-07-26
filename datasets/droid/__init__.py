from typing import Optional

from .dataset import DroidDataset, DroidMixtureDataset
from .dataset_track import DroidTrackDataset

__all__ = [
    "DroidDataset",
    "DroidMixtureDataset",
    "DroidTrackDataset",
    "make_droid_dataset",
]


def make_droid_dataset(
    name: str,
    buffer_path: str,
    shape_meta: dict,
    seq_len: int,
    history_len: int = 1,
    normalize_action: bool = False,
    normalize_lowdim: bool = False,
    val_ratio: float = 0.0,
    video_buffer_path: Optional[str] = None,
    balance_datasets: bool = False,
):
    train_set = DroidDataset(
        name=name,
        buffer_path=buffer_path,
        shape_meta=shape_meta,
        seq_len=seq_len,
        history_len=history_len,
        normalize_lowdim=normalize_lowdim,
        normalize_action=normalize_action,
        val_ratio=val_ratio,
    )
    if video_buffer_path is not None:
        train_set = DroidMixtureDataset(
            base_dataset=train_set,
            video_buffer_path=video_buffer_path,
            balance_datasets=balance_datasets,
        )

    val_set = train_set.get_validation_dataset()
    return train_set, val_set
