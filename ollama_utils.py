from __future__ import annotations

from ollama import Client
from typing import Any, Optional, Dict
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.exceptions import LLMGenerationError, EmbeddingsGenerationError
from config import OLLAMA_BASE_URL

class OllamaLLM(LLMInterface):
    """Ollama LLM implementation for local LLM usage."""
    def __init__(
        self,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize Ollama LLM."""
        super().__init__(model_name, model_params)
        self.client = Client(host=kwargs.get('base_url', OLLAMA_BASE_URL))

    def invoke(self, input: str) -> LLMResponse:
        """Synchronous invocation of Ollama API"""
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=input,
                options=self.model_params or {}
            )
            return LLMResponse(content=response['response'])
        except Exception as e:
            raise LLMGenerationError(f"Ollama generation failed: {str(e)}")

    async def ainvoke(self, input: str) -> LLMResponse:
        """Asynchronous invocation (falls back to sync)"""
        return self.invoke(input)


class OllamaEmbedder(Embedder):
    """Ollama Embeddings implementation for local embedding generation."""
    def __init__(self, model: str = "mistral", **kwargs: Any) -> None:
        """Initialize Ollama Embedder."""
        self.model = model
        self.client = Client(host=kwargs.get('base_url', OLLAMA_BASE_URL))

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embeddings using Ollama's embedding endpoint"""
        try:
            response = self.client.embeddings(
                model=self.model,
                prompt=text,
                options=kwargs
            )
            embedding = response['embedding']
            if not isinstance(embedding, list) or not embedding:
                raise EmbeddingsGenerationError("Invalid embedding format received from Ollama")
            return embedding
        except Exception as e:
            raise EmbeddingsGenerationError(f"Ollama embeddings generation failed: {str(e)}")

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Asynchronous embedding generation (falls back to sync)"""
        return self.embed_query(text, **kwargs)
 