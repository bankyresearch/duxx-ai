"""Retrieval-Augmented Generation (RAG) — document loaders, splitters, embeddings, vector stores, and retrievers."""

from duxx_ai.rag.embeddings import Embedder, LocalEmbedder, OpenAIEmbedder
from duxx_ai.rag.loaders import (
    CSVLoader,
    Document,
    DocumentLoader,
    JSONLLoader,
    TextLoader,
    WebLoader,
)
from duxx_ai.rag.retriever import HybridRetriever, KeywordRetriever, Retriever, VectorRetriever
from duxx_ai.rag.splitters import CharacterSplitter, RecursiveSplitter, TextSplitter, TokenSplitter
from duxx_ai.rag.vectorstore import InMemoryVectorStore, VectorStore

__all__ = [
    "Document", "DocumentLoader", "TextLoader", "CSVLoader", "JSONLLoader", "WebLoader",
    "TextSplitter", "CharacterSplitter", "RecursiveSplitter", "TokenSplitter",
    "Embedder", "OpenAIEmbedder", "LocalEmbedder",
    "VectorStore", "InMemoryVectorStore",
    "Retriever", "VectorRetriever", "KeywordRetriever", "HybridRetriever",
]
