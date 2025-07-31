from pathlib import Path
from typing import List

from fastmcp import FastMCP

from surrealdb_client import SurrealClient
from utils.embeddings import EmbeddingService
from utils.file_loader import FileLoader
from utils.ingestion import IngestionPipeline


class RAGMCPServer:
    mcp = FastMCP("RAG")

    def __init__(
        self,
        client: SurrealClient,
        embedder: EmbeddingService,
        file_loader: FileLoader,
        ingestion_pipeline: IngestionPipeline,
    ):
        self.client = client
        self.embedder = embedder
        self.file_loader = file_loader
        self.pipeline = ingestion_pipeline

    @mcp.tool(name="ingest_directory")
    def ingest_directory(self, directory: str) -> str:
        injest_directory = Path(directory)
        self.pipeline.ingest_directory(injest_directory)
        return f"Ingestion of '{directory}' completed."

    @mcp.tool(name="query")
    def query(self, text: str, top_k: int = 4) -> List[dict]:
        query_vec = self.embedder.encode(text)[0]
        results = self.client.vector_search(text, query_vec, top_k=top_k)

        return [
            {
                "text": node.text.replace("\n", " ")[:200],
            }
            for node in results
        ]
