from pathlib import Path
from typing import List


def list_files(directory: str | Path) -> List[Path]:
    """Return every file in *directory* (non-recursive) as simple filenames."""
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(
            f"Directory '{directory}' does not exist or is not a directory."
        )

    return [f for f in directory.rglob("*") if f.is_file()]
