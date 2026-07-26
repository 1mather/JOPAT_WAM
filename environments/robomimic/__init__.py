"""Factories for RoboMimic and LIBERO evaluation environments."""

import os
from pathlib import Path

from .wrappers import LIBEROEnvWrapper, RoboMimicEnvWrapper

__all__ = ["LIBEROEnvWrapper", "RoboMimicEnvWrapper", "make_robomimic_env"]


def _render_device_id():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_devices:
        return None
    return int(visible_devices.split(",")[0])


def _libero_search_order(dataset_path: str) -> list[str]:
    task_suites = [
        "libero_goal",
        "libero_10",
        "libero_90",
        "libero_spatial",
        "libero_object",
    ]
    for suite in task_suites:
        if suite in dataset_path:
            return [suite] + [item for item in task_suites if item != suite]
    return task_suites


def make_robomimic_env(
    dataset_name,
    dataset_path,
    shape_meta,
    obs_horizon,
    max_episode_length,
    record=False,
):
    """Construct an evaluation environment for a supported dataset family."""
    render_device_id = _render_device_id()

    if "robomimic" in dataset_name:
        from robomimic.utils.env_utils import get_env_class, get_env_type
        from robomimic.utils.file_utils import (
            get_env_metadata_from_dataset,
            get_shape_metadata_from_dataset,
        )
        from robomimic.utils.obs_utils import (
            initialize_obs_utils_with_obs_specs,
        )

        rgb_keys = [
            key
            for key, value in shape_meta["obs"].items()
            if value["type"] == "rgb"
        ]
        low_dim_keys = [
            key
            for key, value in shape_meta["obs"].items()
            if value["type"] == "low_dim"
        ]
        all_obs_keys = rgb_keys + low_dim_keys
        initialize_obs_utils_with_obs_specs(
            {"obs": {"rgb": rgb_keys, "low_dim": low_dim_keys}}
        )

        env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
        env_type = get_env_type(env_meta=env_meta)
        env_class = get_env_class(env_type=env_type)
        dataset_shape_meta = get_shape_metadata_from_dataset(
            dataset_path=dataset_path,
            all_obs_keys=all_obs_keys,
            verbose=True,
        )
        if render_device_id is not None:
            env_meta["env_kwargs"]["render_gpu_device_id"] = render_device_id
        env = env_class(
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=dataset_shape_meta["use_images"],
            use_depth_obs=dataset_shape_meta["use_depths"],
            postprocess_visual_obs=False,
            **env_meta["env_kwargs"],
        )
        return RoboMimicEnvWrapper(
            env,
            all_obs_keys,
            obs_horizon,
            max_episode_length,
            record=record,
        )

    if "libero" in dataset_name:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl_filename = (
            Path(dataset_path)
            .name.replace("_demo.zarr", ".bddl")
            .replace("_demo.hdf5", ".bddl")
        )
        search_order = _libero_search_order(dataset_path)
        bddl_file = None
        for task_suite in search_order:
            candidate = (
                Path(get_libero_path("bddl_files"))
                / task_suite
                / bddl_filename
            )
            if candidate.exists():
                bddl_file = candidate
                break
        if bddl_file is None:
            raise FileNotFoundError(
                f"Could not find {bddl_filename!r} in the LIBERO task suites "
                f"{search_order}."
            )

        env_kwargs = {
            "bddl_file_name": str(bddl_file),
            "camera_heights": 128,
            "camera_widths": 128,
        }
        if render_device_id is not None:
            env_kwargs["render_gpu_device_id"] = render_device_id
        env = OffScreenRenderEnv(**env_kwargs)
        return LIBEROEnvWrapper(
            env,
            list(shape_meta["obs"].keys()),
            obs_horizon,
            max_episode_length,
            record=record,
        )

    raise NotImplementedError(f"Unsupported environment: {dataset_name}")
