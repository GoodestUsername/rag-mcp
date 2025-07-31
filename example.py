import asyncio
import os
from pathlib import Path
from typing import Sequence, Union

from dotenv import load_dotenv
from llama_index.core.schema import Document
from surrealdb import AsyncSurreal
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection

from surrealdb_client import SurrealClient
from utils.embeddings import EmbeddingService
from utils.file_loader import FileLoader
from utils.fs import list_files

load_dotenv()
DB_URL = os.getenv("SDB_URL", "")
DB_USER = os.getenv("SDB_USER", "")
DB_PASS = os.getenv("SDB_PASS", "")
NS = os.getenv("NS", "")
DB = os.getenv("DB", "")
FILENAME_TABLE = os.getenv("FILENAME_TABLE", "")
FILE_CHUNK_TABLE = os.getenv("FILE_CHUNK_TABLE", "")

FILES_DIR = Path(os.getenv("FILES_DIR", ""))
EMBED_MODEL = os.getenv("EMBED_MODEL", "")
CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "")
file_loader = FileLoader()
embedder = EmbeddingService(EMBED_MODEL, cache_dir=CACHE_DIR)


def _create_db_connection() -> (
    Union[AsyncWsSurrealConnection, AsyncHttpSurrealConnection]
):
    client = AsyncSurreal(DB_URL)
    asyncio.run(client.signin({"username": DB_USER, "password": DB_PASS}))
    asyncio.run(client.use(NS, DB))
    return client


def _documents_from_file(path: Path) -> Sequence[Document]:
    docs = file_loader.load(path)
    embeddings = embedder.encode([doc.text for doc in docs])
    for doc, emb in zip(docs, embeddings, strict=True):  # type: ignore[arg-type]
        doc.embedding = emb  # type: ignore[attr-defined]
    return docs


def ingest_directory(directory: Path, client: SurrealClient) -> None:
    files = list_files(directory)
    filenames = [os.path.basename(file) for file in files]
    missing = client.files_not_uploaded(filenames)

    for filename in missing:
        docs = _documents_from_file(directory / filename)
        client.add_file(filename, docs)
        print(f"✓ Ingested {filename} ({len(docs)} chunks)")


def demo_query(client: SurrealClient, query: str, *, k: int = 4) -> None:
    print("\nRunning demo query …")
    query_vec = embedder.encode(query).tolist()[0]
    results = client.vector_search(query, query_vec, top_k=k)

    print("\nTop results:\n-----------")
    for node in results:
        snippet = node.text.replace("\n", " ")[:120]
        print(snippet)


if __name__ == "__main__":
    if not FILES_DIR.exists():
        raise SystemExit(f"Files directory '{FILES_DIR}' not found.")

    surreal = SurrealClient(
        filename_table=FILENAME_TABLE,
        vector_table=FILE_CHUNK_TABLE,
        conn=_create_db_connection(),
    )

    ingest_directory(FILES_DIR, surreal)
    demo_query(surreal, "Panda?")
