import glob
from os.path import expanduser, expandvars
from typing import Union

from omegaconf.listconfig import ListConfig


def glob_all(globs: Union[str, list[str], ListConfig]) -> list[str]:
    """Expand one or more user- and environment-aware glob patterns."""
    if isinstance(globs, str):
        globs = [globs]
    elif isinstance(globs, ListConfig):
        globs = list(globs)
    if not isinstance(globs, list):
        raise TypeError("globs must be a string or a list of strings.")

    files = []
    for glob_str in globs:
        glob_str = expandvars(expanduser(glob_str))
        files.extend(glob.glob(glob_str))

    return sorted(files)
