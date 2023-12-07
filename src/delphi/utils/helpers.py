"""
Utility functions for directory and file management, and terminal text coloring.

This module provides helper functions to:

- Create a new subdirectory if it does not exist.
- Print text in red color in the terminal.
- Check and return the absolute path of a given file.
"""
import os
from pathlib import Path
import re
import subprocess
import sys


def get_git_commit_hash() -> str:
    """Gets the current git commit hash for future use of a model.

    Returns:
        str: Git commit hash
    """
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        print(e)
        git_hash = "Unknown"
    return git_hash


def create_subdir(dir_name: str | Path, sub_dir: str) -> Path:
    """Creates a new directory 'sub_dir' in 'dir' if not exists already.

    Args:
        dir (str | Path): Parent directory.
        sub_dir (str): Name of the subdirectory.

    Returns:
        Path: Full path to sub-directory.
    """
    if not isinstance(dir_name, Path):
        dir_name = Path(dir_name)
    # cleaning up sub_dir_name (removing .,: etc)
    sub_dir = re.sub(r"[^A-Za-z0-9]+", "_", sub_dir)
    subdir = dir_name.joinpath(sub_dir)
    # create the directory if not exist already
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def is_path(path_str: str) -> Path:
    """Checks and returns the absolute path of a given file.

    Args:
        path (str): The path of the file.

    Returns:
        filepath (Path): The absolute path of the file.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
    """
    # check for path - assuming absolute path was given
    filepath = Path(path_str)
    if not filepath.exists():
        # assuming path was given relative to cwd
        rootdir = Path(os.getcwd())
        filepath = rootdir.joinpath(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"{path_str} not found. Define relative to project directory\
                or as absolute path in argument passing."
        )
    return filepath


class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
