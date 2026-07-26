from typing import Any


def unflatten_obs(x: dict[str, Any]) -> dict[str, Any]:
    """Move entries named ``obs.<key>`` into a nested ``obs`` dictionary."""
    obs = {}
    to_delete = []
    for key, value in x.items():
        if key.startswith("obs."):
            new_key = key[4:]
            if new_key in obs:
                raise ValueError(f"Duplicate observation key: {new_key}.")
            obs[new_key] = value
            to_delete.append(key)
    for key in to_delete:
        del x[key]
    x["obs"] = obs
    return x
