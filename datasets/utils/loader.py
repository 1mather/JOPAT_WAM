import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def custom_collate_fn(batch):
    """Collate tensors recursively while preserving string-like arrays."""
    if not batch:
        return {}

    elem = batch[0]
    if isinstance(elem, dict):
        collated = {}
        for key in elem.keys():
            values = [d[key] for d in batch]
            if (
                isinstance(values[0], np.ndarray)
                and values[0].dtype.kind in {"U", "S", "O"}
            ):
                collated[key] = values
            elif isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            elif isinstance(values[0], dict):
                collated[key] = custom_collate_fn(values)
            elif isinstance(values[0], (int, float, np.number)):
                collated[key] = torch.tensor(values)
            else:
                collated[key] = values
        return collated
    return torch.utils.data.dataloader.default_collate(batch)


def make_distributed_data_loader(
    train_set: Dataset,
    val_set: Dataset,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = True,
):
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=custom_collate_fn,
    )

    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=custom_collate_fn,
    )

    return train_loader, val_loader
