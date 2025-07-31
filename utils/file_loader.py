from pathlib import Path
from typing import Sequence, Union

import pymupdf4llm
from llama_index.core.schema import Document


class FileLoader:
    def __init__(self) -> None:
        self._reader = pymupdf4llm.LlamaMarkdownReader()

    def load(self, path: Union[str, Path]) -> Sequence[Document]:
        if not isinstance(path, (str, Path)):
            raise TypeError("`path` must be a `str` or `Path` instance.")
        return self._reader.load_data(str(path))
