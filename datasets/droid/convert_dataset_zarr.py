import argparse
import multiprocessing as mp
import os

# Prevent conversion workers from reserving accelerator memory.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from tqdm import tqdm

import tensorflow_datasets as tfds

from ..utils.buffer import CompressedTrajectoryBuffer
from .utils import euler_angles_to_rot_6d


shape_meta = {
    "obs": {
        "exterior_image_1_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "exterior_image_2_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "wrist_image_left": {
            "shape": (180, 320, 3),
            "type": "rgb",
        },
        "cartesian_position": {
            "shape": (6,),
            "type": "low_dim",
        },
        "gripper_position": {
            "shape": (1,),
            "type": "low_dim",
        },
    },
    "action": {
        "shape": (10,),
    },
}

rgb_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "rgb"]
lowdim_keys = [k for k, v in shape_meta["obs"].items() if v["type"] == "low_dim"]


def _load_episode_impl(episode, zero_action: bool):
    obs_dict = {k: [] for k in shape_meta["obs"].keys()}
    actions = []
    for step in episode:
        obs = step["observation"]
        for k in rgb_keys:
            obs_dict[k].append(obs[k].numpy())
        obs_dict["cartesian_position"].append(obs["cartesian_position"].numpy())
        obs_dict["gripper_position"].append(obs["gripper_position"].numpy())

        action_dict = step["action_dict"]
        cartesian_velocity = action_dict["cartesian_velocity"].numpy()
        xyz, euler = cartesian_velocity[:3], cartesian_velocity[3:6]
        rot6d = euler_angles_to_rot_6d(torch.tensor(euler)).numpy()
        grippers = action_dict["gripper_position"].numpy()
        action = np.concatenate([xyz, rot6d, grippers])
        actions.append(action)

    episode_dict = {"obs." + k: np.stack(v) for k, v in obs_dict.items()}
    action_arr = np.stack(actions).astype(np.float32)
    if zero_action:
        action_arr[...] = 0.0
    episode_dict["action"] = action_arr
    return episode_dict


def preprocess_episode(episode):
    """Preprocess episode by removing variant tensors and standardizing data types."""
    processed_episode = []
    for step in episode["steps"]:
        processed_step = {
            "observation": step["observation"],
            "action_dict": step["action_dict"],
        }
        processed_episode.append(processed_step)
    return processed_episode


def episode_loader(queue, buffer_args, *, zero_action: bool):
    """Process episodes from input queue and put results in output queue."""
    pid = mp.current_process().name
    print(f"Starting episode loader process {pid}")

    buffer = CompressedTrajectoryBuffer(**buffer_args)

    while True:
        episode = queue.get()
        if episode is None:
            print(f"Episode loader {pid} received a termination signal")
            break
        buffer.add_episode(_load_episode_impl(episode, zero_action=zero_action))


def sample_episodes(
    raw_dataset,
    num_episodes: int,
    filter_key=None,
    except_key=None,
    seed: int = 0,
):
    """Select and deterministically shuffle a fixed number of episodes."""
    episodes = []
    for episode in raw_dataset:
        folderpath = (
            episode["episode_metadata"]["recording_folderpath"].numpy().decode("UTF-8")
        )
        if filter_key and filter_key not in folderpath:
            continue
        if except_key and except_key in folderpath:
            continue
        episodes.append(episode)
        if len(episodes) >= num_episodes:
            break
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    return episodes


def main(args):
    if args.num_action_free_episodes and not args.buffer_path_action_free:
        raise ValueError(
            "--buffer_path_action_free is required when "
            "--num_action_free_episodes is positive."
        )
    raw_dataset = tfds.load(args.data_name, data_dir=args.data_dir, split="train")
    total_needed = args.num_action_episodes + args.num_action_free_episodes
    episodes = sample_episodes(
        raw_dataset,
        num_episodes=total_needed,
        filter_key=args.filter_key,
        except_key=args.except_key,
        seed=args.seed,
    )
    if len(episodes) < total_needed:
        raise RuntimeError(
            f"Only found {len(episodes)} episodes after filtering, but need {total_needed}."
        )
    action_episodes = episodes[: args.num_action_episodes]
    action_free_episodes = episodes[args.num_action_episodes : total_needed]

    metadata = {}
    for key, meta in shape_meta["obs"].items():
        metadata[f"obs.{key}"] = {
            "shape": meta["shape"],
            "dtype": np.uint8 if meta["type"] == "rgb" else np.float32,
        }
    metadata["action"] = {"shape": shape_meta["action"]["shape"], "dtype": np.float32}

    def write_buffer(*, selected_episodes, buffer_path: str, zero_action: bool):
        capacity = sum(len(episode["steps"]) for episode in selected_episodes)
        print(f"Buffer capacity ({buffer_path}): {capacity}")

        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path,
            metadata=metadata,
            capacity=capacity,
        )
        if buffer.restored:
            print(f"Buffer restored from disk: {buffer_path}")
            return

        context = mp.get_context("spawn")
        queue = context.Queue(maxsize=args.num_workers * 2)
        lock = context.Lock()
        buffer_args = {
            "storage_path": buffer_path,
            "metadata": metadata,
            "capacity": capacity,
            "lock": lock,
        }

        episode_loader_processes = []
        for i in range(args.num_workers):
            p = context.Process(
                target=episode_loader,
                args=(queue, buffer_args),
                kwargs={"zero_action": zero_action},
                name=f"EpisodeLoaderProcess-{i}",
            )
            p.start()
            episode_loader_processes.append(p)

        for episode in tqdm(
            selected_episodes,
            desc=f"Preprocessing ({os.path.basename(buffer_path)})",
        ):
            processed_episode = preprocess_episode(episode)
            queue.put(processed_episode)

        for _ in range(args.num_workers):
            queue.put(None)

        for p in episode_loader_processes:
            p.join()

    write_buffer(
        selected_episodes=action_episodes,
        buffer_path=args.buffer_path_action,
        zero_action=False,
    )
    if action_free_episodes:
        write_buffer(
            selected_episodes=action_free_episodes,
            buffer_path=args.buffer_path_action_free,
            zero_action=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DROID episodes into compressed Zarr buffers."
    )
    parser.add_argument("--data_name", type=str, default="droid")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--buffer_path_action",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--buffer_path_action_free",
        type=str,
        default=None,
    )
    parser.add_argument("--num_action_episodes", type=int, default=2000)
    parser.add_argument("--num_action_free_episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--except_key", type=str, default=None)
    args = parser.parse_args()
    main(args)
