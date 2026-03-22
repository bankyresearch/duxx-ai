"""Retrieval-Augmented Generation (RAG) — document loaders, splitters, embeddings, vector stores, and retrievers."""

from duxx_ai.rag.loaders import Document, DocumentLoader, TextLoader, CSVLoader, JSONLLoader, WebLoader
from duxx_ai.rag.splitters import TextSplitter, CharacterSplitter, RecursiveSplitter, TokenSplitter
from duxx_ai.rag.embeddings import Embedder, OpenAIEmbedder, LocalEmbedder
from duxx_ai.rag.vectorstore import VectorStore, InMemoryVectorStore
from duxx_ai.rag.retriever import Retriever, VectorRetriever, KeywordRetriever, HybridRetriever

__all__ = [
    "Document", "DocumentLoader", "TextLoader", "CSVLoader", "JSONLLoader", "WebLoader",
    "TextSplitter", "CharacterSplitter", "RecursiveSplitter", "TokenSplitter",
    "Embedder", "OpenAIEmbedder", "LocalEmbedder",
    "VectorStore", "InMemoryVectorStore",
    "Retriever", "VectorRetriever", "KeywordRetriever", "HybridRetriever",
]
