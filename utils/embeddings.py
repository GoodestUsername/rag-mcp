from typing import List

import torch
from sentence_transformers import SentenceTransformer

# Default paths can be overridden through environment variables for flexibility


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: str,
        use_cuda: bool = True,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and not torch.cuda.is_available():
            print("CUDA unavailable – falling back to CPU for embeddings.")

        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_dir,
                trust_remote_code=True,
            )
        return self._model

    def encode(self, sentences: List[str] | str, *, batch_size: int = 32, **kwargs):  # type: ignore[override]
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.model.encode(
            sentences, batch_size=batch_size, convert_to_tensor=False, **kwargs
        )
