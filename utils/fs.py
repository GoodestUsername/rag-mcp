from fnmatch import fnmatch
from pathlib import Path
from typing import List


def list_files(directory: str | Path, ignore_rules: List[str] = []) -> List[Path]:
    """
    Recursively list all file paths in a directory and its subdirectories,
    excluding files and directories that match any of the given fnmatch style patterns.

    Args:
        directory (str | Path): The root directory to search in.
        ignore_rules (List[str], optional): A list of Unix shell-style patterns
            (e.g. '*.log', 'build') used to exclude matching files and directories.
            Defaults to an empty list.

    Returns:
        List[Path]: A list of Path objects representing all matched files.

    Raises:
        FileNotFoundError: If the provided path does not exist or is not a directory.

    Example:
        >>> list_files("src", ignore_rules=["*.pyc", "__pycache__"])
        [PosixPath('src/main.py'), PosixPath('src/utils/helper.py')]
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist or is not a directory."
        )

    matches: List[Path] = []

    for dirpath, dirnames, filenames in directory.walk():
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not any(fnmatch(dirname, rule) for rule in ignore_rules)
        ]

        matches.extend(
            [
                dirpath / filename
                for filename in filenames
                if not any(fnmatch(filename, rule) for rule in ignore_rules)
            ]
        )

    return matches
