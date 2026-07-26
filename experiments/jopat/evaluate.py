"""Checkpoint rollout evaluation for JOPAT."""

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from experiments.jopat.train import collect_rollouts, rollout_is_supported
from experiments.utils import set_seed
from models.jopat.checkpoint import load_from_checkpoint


@hydra.main(version_base=None, config_path="../../configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    if not torch.cuda.is_available():
        raise RuntimeError("JOPAT rollout evaluation requires a CUDA device.")
    if not rollout_is_supported(config):
        raise ValueError(
            "Evaluation requires one LIBERO HDF5 file and rollout metadata."
        )

    set_seed(int(config.seed))
    device = torch.device("cuda", 0)
    model = instantiate(config.model).to(device)
    model, load_info = load_from_checkpoint(model, config.checkpoint_path)
    if float(load_info["loading_ratio"]) < 1.0:
        raise RuntimeError(
            "The checkpoint is not fully compatible with this model configuration."
        )

    success_rate, _, _ = collect_rollouts(config, model, device)
    print(f"Rollout success rate: {success_rate:.2%}")


if __name__ == "__main__":
    main()
