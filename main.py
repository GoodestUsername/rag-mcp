import asyncio
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Union

import anyio
import typer
from dotenv import load_dotenv
from surrealdb import AsyncSurreal
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection

from mcp_server import RAGMCPServer
from surrealdb_client import SurrealClient
from utils.embeddings import EmbeddingService
from utils.file_loader import FileLoader
from utils.ingestion import IngestionPipeline

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


class HTTPTransportTypes(Enum):
    http = "http"
    streamable_http = "streamable-http"
    sse = "sse"


def _create_db_connection() -> (
    Union[AsyncWsSurrealConnection, AsyncHttpSurrealConnection]
):
    client = AsyncSurreal(DB_URL)
    asyncio.run(client.signin({"username": DB_USER, "password": DB_PASS}))
    asyncio.run(client.use(NS, DB))
    return client


def main():
    default_path = os.environ.get("DEFAULT_PATH", "mcp")
    surreal_client = SurrealClient(
        conn=_create_db_connection(),
        filename_table=FILENAME_TABLE,
        vector_table=FILE_CHUNK_TABLE,
    )
    file_loader = FileLoader()
    embedder = EmbeddingService(EMBED_MODEL, cache_dir=CACHE_DIR)
    ingestion_pipeline = IngestionPipeline(
        embedder,
        file_loader,
        surreal_client,
    )
    rag_server = RAGMCPServer(surreal_client, embedder, file_loader, ingestion_pipeline)
    app = typer.Typer()

    @app.command()
    def stdio(show_banner: bool = True):
        anyio.run(
            partial(
                rag_server.mcp.run_stdio_async,
                show_banner=show_banner,
            )
        )

    @app.command()
    def http(
        show_banner: bool = True,
        transport: HTTPTransportTypes = HTTPTransportTypes.http,
        host: str | None = None,
        port: int | None = None,
        log_level: str | None = None,
        path: str | None = None,
        stateless_http: bool | None = None,
    ):
        anyio.run(
            partial(
                rag_server.mcp.run_http_async,
                show_banner=show_banner,
                transport=transport.value,
                host=host,
                port=port,
                log_level=log_level,
                path=f"/{path if path is not None else default_path}",
                stateless_http=stateless_http,
            )
        )

    app()


if __name__ == "__main__":
    main()
