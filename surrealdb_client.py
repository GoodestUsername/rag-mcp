import asyncio
import logging
import uuid
from typing import List, Sequence, Union

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from pydantic import BaseModel
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection
from surrealdb.data.types.record_id import RecordID

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SurrealClient(BaseModel):
    filename_table: str
    vector_table: str
    conn: Union[AsyncWsSurrealConnection, AsyncHttpSurrealConnection]
    model_config = {"arbitrary_types_allowed": True}

    def files_not_uploaded(self, filenames: List[str]) -> list[str]:
        res = asyncio.run(
            self.conn.query(
                f"""
                array::difference(return (
                    SELECT filename
                    FROM {self.filename_table}
                    WHERE filename in $file_names).filename, $file_names
                )""",
                {"file_names": filenames},
            )
        )
        return res  # type: ignore

    def add_file(self, filename: str, nodes: Sequence[BaseNode]):
        file_id = RecordID(self.filename_table, uuid.uuid4().hex)
        file_chunks = [
            {
                # "id": f"{self.vector_table}:{i}",
                "file": file_id,
                "text": n.get_content(metadata_mode=MetadataMode.ALL),
                "embedding": n.embedding,
            }
            for n in nodes
        ]
        asyncio.run(
            self.conn.query(
                f"""
                let $file = create only {self.filename_table} content {{
                    id: {file_id},
                    filename: "{filename}",
                    file_chunks: []
                }};
                let $file_chunk_ids = insert into {self.vector_table} $filechunks return value id;

                update $file set file_chunks = $file_chunk_ids;
            """,
                {"filechunks": file_chunks},
            )
        )

    def delete_file(self, filename: str):
        asyncio.run(
            self.conn.query(
                f"""
                let $del_file = delete only {self.filename_table} where filename = $file_name return before;
                delete {self.vector_table} where file = $del_file.id;
            """,
                {"file_name": filename},
            )
        )

    def vector_search(
        self, query: str, query_embedding: List[float], top_k: int
    ) -> Sequence[TextNode]:
        logger.debug("querying nodes", query)
        surql = f"""
        SELECT id, text, vector::distance::knn() AS score
        FROM {self.vector_table}
        WHERE embedding <|{top_k},64|> $vec
        ORDER BY score;
        """
        res = asyncio.run(self.conn.query(surql, {"vec": query_embedding}))
        if len(res) == 0 or res is None:
            print("Result none")
            return []
        print("resutl not none")

        nodes = [TextNode(id=r["id"].id, text=r["text"], score=r["score"]) for r in res]
        print(nodes)
        return nodes
