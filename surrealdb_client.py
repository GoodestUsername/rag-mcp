import asyncio
import logging
import uuid
from typing import List, Sequence, TypedDict, Union

from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from pydantic import BaseModel
from surrealdb.connections.async_http import AsyncHttpSurrealConnection
from surrealdb.connections.async_ws import AsyncWsSurrealConnection
from surrealdb.data.types.record_id import RecordID

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FileChunk(TypedDict):
    id: RecordID
    text: str
    embedding: List[float]


class File(TypedDict):
    id: RecordID
    filename: str
    file_chunks: List[FileChunk]


class SearchRankingFileChunk(FileChunk):
    score: float


def rrf(weight, rank, rrf_k):
    return weight * (1.0 / (rrf_k + rank))


def rrf_reorder(
    vector_results: List[SearchRankingFileChunk],
    full_text_results: List[SearchRankingFileChunk],
    vector_search_weight: float = 1.0,
    full_text_search_weight: float = 1.0,
    rrf_k: int = 60,
) -> List[SearchRankingFileChunk]:
    rrf_scores = {}

    for rank, document in enumerate(vector_results, start=1):
        rrf_scores[document["id"].id] = {
            "document": document,
            "rrf_score": rrf(vector_search_weight, rrf_k, rank),
        }

    for rank, document in enumerate(full_text_results, start=1):
        if document["id"].id in rrf_scores:
            rrf_scores[document["id"].id]["rrf_score"] += rrf(
                full_text_search_weight, rrf_k, rank
            )
        else:
            rrf_scores[document["id"].id] = {
                "document": document,
                "rrf_score": rrf(vector_search_weight, rrf_k, rank),
            }

    combined = []
    combined.extend(vector_results)
    combined.extend(full_text_results)
    order = sorted(rrf_scores.items(), key=lambda x: -x[-1]["rrf_score"])
    return [document[-1]["document"] for document in order]


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
        # https://github.com/surrealdb/examples/blob/main/hybrid-search/hybrid-search.surql
        query_string = f"""
        BEGIN TRANSACTION;

        LET $vector_search = (
            SELECT id, text, vector::distance::knn() AS score
            FROM {self.vector_table}
            WHERE embedding <|{top_k},64|> $vec
            ORDER BY score DESC
            LIMIT 10
        );
        
        LET $text_search = (
            SELECT *,
            search::highlight("**", "**", 1) AS body,
            search::highlight("##", "", 0) AS title,
            search::score(0) + search::score(1) AS score
            FROM {self.vector_table}
            WHERE text @0@ $full_text
            OR text @1@ $full_text
            ORDER BY score DESC
            LIMIT 10
        );
        
        RETURN {{
          vector_results: $vector_search,
          text_results: $text_search
        }};
        
        COMMIT TRANSACTION;
        """

        res: dict[str, List[SearchRankingFileChunk]] = asyncio.run(
            self.conn.query(query_string, {"vec": query_embedding, "full_text": query})
        )  # type: ignore
        reordered = rrf_reorder(res["vector_results"], res["text_results"], 1, 1, 60)
        if reordered is None or len(reordered) == 0:
            return []

        nodes = [TextNode(id=r["id"], text=r["text"]) for r in reordered]
        return nodes
