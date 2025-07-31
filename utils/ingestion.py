import os
from pathlib import Path
from typing import Sequence

from llama_index.core.schema import Document

from surrealdb_client import SurrealClient
from utils.embeddings import EmbeddingService
from utils.file_loader import FileLoader
from utils.fs import list_files


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbeddingService,
        file_loader: FileLoader,
        client: SurrealClient,
    ):
        self._loader = file_loader
        self._embedder = embedder
        self._client = client

    def ingest_file(self, path: Path):
        docs: Sequence[Document] = self._loader.load(path)
        embeddings = self._embedder.encode([d.text for d in docs])
        for doc, emb in zip(docs, embeddings, strict=True):
            doc.embedding = emb
        self._client.add_file(path.name, docs)

    def ingest_directory(self, directory: Path):
        files = list_files(directory)
        filename_path_map = {os.path.basename(file): file for file in files}
        missing = self._client.files_not_uploaded([*filename_path_map])

        for filename in missing:
            self.ingest_file(filename_path_map[filename])
