"""Data source connectors — load training data from any source.

Supports: HuggingFace Datasets, AWS S3, Google Cloud Storage, Azure Blob,
Google Drive, SQL/NoSQL databases, and local files (JSONL, CSV, Excel, Parquet, JSON, PDF).

All connectors implement DataConnector with load() and test_connection().
Optional dependencies are lazy-imported for graceful degradation.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DataConnector(ABC):
    """Base class for all data source connectors."""

    @abstractmethod
    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        """Load data and return a list of dicts."""
        ...

    @abstractmethod
    def test_connection(self) -> dict[str, Any]:
        """Test connectivity. Returns {"connected": bool, "details": ...}."""
        ...

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Return the source type identifier."""
        ...


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HuggingFace Datasets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HuggingFaceConnector(DataConnector):
    """Load datasets from HuggingFace Hub.

    Requires: pip install datasets
    Usage: HuggingFaceConnector("tatsu-lab/alpaca", split="train")
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        token: str | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.token = token or os.environ.get("HF_TOKEN", "")

    @property
    def source_type(self) -> str:
        return "huggingface"

    def test_connection(self) -> dict[str, Any]:
        try:
            from datasets import load_dataset_builder  # type: ignore
            builder = load_dataset_builder(self.dataset_name, self.subset, token=self.token or None)
            info = builder.info
            return {
                "connected": True,
                "dataset": self.dataset_name,
                "description": (info.description or "")[:200],
                "features": str(info.features)[:300] if info.features else "",
                "size": info.dataset_size,
                "splits": list(info.splits.keys()) if info.splits else [],
            }
        except ImportError:
            return {"connected": False, "error": "Install: pip install datasets"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError:
            raise ImportError("HuggingFace datasets not installed. Run: pip install datasets")

        kwargs: dict[str, Any] = {"split": self.split}
        if self.subset:
            kwargs["name"] = self.subset
        if self.token:
            kwargs["token"] = self.token

        dataset = load_dataset(self.dataset_name, **kwargs)

        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        return [dict(row) for row in dataset]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  AWS S3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class S3Connector(DataConnector):
    """Load data from AWS S3. Supports JSONL, CSV, Parquet, JSON.

    Requires: pip install boto3
    """

    def __init__(
        self,
        bucket: str,
        key: str,
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
    ) -> None:
        self.bucket = bucket
        self.key = key
        self.region = region
        self.access_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "")
        self.secret_key = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "")

    @property
    def source_type(self) -> str:
        return "s3"

    def _get_client(self) -> Any:
        try:
            import boto3  # type: ignore
        except ImportError:
            raise ImportError("boto3 not installed. Run: pip install boto3")
        kwargs: dict[str, Any] = {"region_name": self.region}
        if self.access_key and self.secret_key:
            kwargs["aws_access_key_id"] = self.access_key
            kwargs["aws_secret_access_key"] = self.secret_key
        return boto3.client("s3", **kwargs)

    def test_connection(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            client.head_object(Bucket=self.bucket, Key=self.key)
            return {"connected": True, "bucket": self.bucket, "key": self.key}
        except ImportError as e:
            return {"connected": False, "error": str(e)}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        import tempfile
        client = self._get_client()
        ext = Path(self.key).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            client.download_file(self.bucket, self.key, tmp.name)
            return _load_file_by_extension(tmp.name, ext, max_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Google Cloud Storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GCSConnector(DataConnector):
    """Load data from Google Cloud Storage.

    Requires: pip install google-cloud-storage
    """

    def __init__(self, bucket: str, blob_name: str, credentials_path: str = "") -> None:
        self.bucket_name = bucket
        self.blob_name = blob_name
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")

    @property
    def source_type(self) -> str:
        return "gcs"

    def _get_client(self) -> Any:
        try:
            from google.cloud import storage  # type: ignore
        except ImportError:
            raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")
        if self.credentials_path:
            return storage.Client.from_service_account_json(self.credentials_path)
        return storage.Client()

    def test_connection(self) -> dict[str, Any]:
        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.blob_name)
            return {"connected": blob.exists(), "bucket": self.bucket_name, "blob": self.blob_name}
        except ImportError as e:
            return {"connected": False, "error": str(e)}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        import tempfile
        client = self._get_client()
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(self.blob_name)
        ext = Path(self.blob_name).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            return _load_file_by_extension(tmp.name, ext, max_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Azure Blob Storage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AzureBlobConnector(DataConnector):
    """Load data from Azure Blob Storage.

    Requires: pip install azure-storage-blob
    """

    def __init__(self, container: str, blob_name: str, connection_string: str = "") -> None:
        self.container = container
        self.blob_name = blob_name
        self.connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")

    @property
    def source_type(self) -> str:
        return "azure"

    def test_connection(self) -> dict[str, Any]:
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
            client = BlobServiceClient.from_connection_string(self.connection_string)
            container = client.get_container_client(self.container)
            blob = container.get_blob_client(self.blob_name)
            props = blob.get_blob_properties()
            return {"connected": True, "container": self.container, "blob": self.blob_name, "size": props.size}
        except ImportError:
            return {"connected": False, "error": "Install: pip install azure-storage-blob"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        import tempfile
        try:
            from azure.storage.blob import BlobServiceClient  # type: ignore
        except ImportError:
            raise ImportError("azure-storage-blob not installed. Run: pip install azure-storage-blob")

        client = BlobServiceClient.from_connection_string(self.connection_string)
        blob_client = client.get_blob_client(self.container, self.blob_name)
        ext = Path(self.blob_name).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            with open(tmp.name, "wb") as f:
                f.write(blob_client.download_blob().readall())
            return _load_file_by_extension(tmp.name, ext, max_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Google Drive
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GoogleDriveConnector(DataConnector):
    """Load data from Google Drive via file ID.

    Requires: pip install google-api-python-client google-auth
    """

    def __init__(self, file_id: str, credentials_path: str = "") -> None:
        self.file_id = file_id
        self.credentials_path = credentials_path

    @property
    def source_type(self) -> str:
        return "google_drive"

    def test_connection(self) -> dict[str, Any]:
        try:
            from googleapiclient.discovery import build  # type: ignore
            from google.oauth2 import service_account  # type: ignore
            creds = service_account.Credentials.from_service_account_file(self.credentials_path)
            service = build("drive", "v3", credentials=creds)
            file_meta = service.files().get(fileId=self.file_id).execute()
            return {"connected": True, "name": file_meta.get("name"), "mimeType": file_meta.get("mimeType")}
        except ImportError:
            return {"connected": False, "error": "Install: pip install google-api-python-client google-auth"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        import tempfile
        try:
            from googleapiclient.discovery import build  # type: ignore
            from google.oauth2 import service_account  # type: ignore
        except ImportError:
            raise ImportError("Install: pip install google-api-python-client google-auth")

        creds = service_account.Credentials.from_service_account_file(self.credentials_path)
        service = build("drive", "v3", credentials=creds)
        file_meta = service.files().get(fileId=self.file_id).execute()
        name = file_meta.get("name", "download.jsonl")
        ext = Path(name).suffix.lower()

        content = service.files().get_media(fileId=self.file_id).execute()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            return _load_file_by_extension(tmp.name, ext, max_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Database (SQL + MongoDB)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DatabaseConnector(DataConnector):
    """Load data from SQL databases (PostgreSQL, MySQL, SQLite) or MongoDB.

    SQL requires: pip install sqlalchemy
    MongoDB requires: pip install pymongo
    """

    def __init__(
        self,
        connection_string: str,
        query: str = "",
        db_type: str = "sql",  # sql | mongodb
        collection: str = "",
    ) -> None:
        self.connection_string = connection_string
        self.query = query
        self.db_type = db_type
        self.collection = collection

    @property
    def source_type(self) -> str:
        return f"database_{self.db_type}"

    def test_connection(self) -> dict[str, Any]:
        try:
            if self.db_type == "mongodb":
                from pymongo import MongoClient  # type: ignore
                client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
                client.server_info()
                return {"connected": True, "type": "mongodb"}
            else:
                from sqlalchemy import create_engine, text  # type: ignore
                engine = create_engine(self.connection_string)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                return {"connected": True, "type": "sql"}
        except ImportError as e:
            return {"connected": False, "error": str(e)}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        if self.db_type == "mongodb":
            return self._load_mongodb(max_samples)
        return self._load_sql(max_samples)

    def _load_sql(self, max_samples: int | None) -> list[dict[str, Any]]:
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except ImportError:
            raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy")

        engine = create_engine(self.connection_string)
        query = self.query
        if max_samples and "LIMIT" not in query.upper():
            query += f" LIMIT {max_samples}"

        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def _load_mongodb(self, max_samples: int | None) -> list[dict[str, Any]]:
        try:
            from pymongo import MongoClient  # type: ignore
        except ImportError:
            raise ImportError("pymongo not installed. Run: pip install pymongo")

        client = MongoClient(self.connection_string)
        db = client.get_default_database()
        collection = db[self.collection]
        query = json.loads(self.query) if self.query else {}
        cursor = collection.find(query)
        if max_samples:
            cursor = cursor.limit(max_samples)

        results = []
        for doc in cursor:
            doc.pop("_id", None)
            results.append(doc)
        return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Local File System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LocalFileConnector(DataConnector):
    """Load data from local files: JSONL, CSV, TSV, Excel, Parquet, JSON, TXT, PDF."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)

    @property
    def source_type(self) -> str:
        return "local"

    def test_connection(self) -> dict[str, Any]:
        if self.path.exists():
            size = self.path.stat().st_size
            return {"connected": True, "path": str(self.path), "size_bytes": size, "format": self.path.suffix}
        return {"connected": False, "error": f"File not found: {self.path}"}

    def load(self, max_samples: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        return _load_file_by_extension(str(self.path), self.path.suffix.lower(), max_samples)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  File Format Loaders (shared utility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_file_by_extension(path: str, ext: str, max_samples: int | None = None) -> list[dict[str, Any]]:
    """Load a file based on its extension, returning a list of dicts."""
    data: list[dict[str, Any]] = []

    if ext in (".jsonl", ".ndjson"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
                    if max_samples and len(data) >= max_samples:
                        break

    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            data = raw
        elif isinstance(raw, dict):
            # Try common nested keys
            for key in ("data", "rows", "items", "examples", "samples", "train"):
                if key in raw and isinstance(raw[key], list):
                    data = raw[key]
                    break
            if not data:
                data = [raw]

    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data.append(dict(row))
                if max_samples and len(data) >= max_samples:
                    break

    elif ext in (".xlsx", ".xls"):
        try:
            import openpyxl  # type: ignore
            wb = openpyxl.load_workbook(path, read_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            if rows:
                headers = [str(h or f"col_{i}") for i, h in enumerate(rows[0])]
                for row in rows[1:]:
                    data.append(dict(zip(headers, [str(v) if v is not None else "" for v in row])))
                    if max_samples and len(data) >= max_samples:
                        break
            wb.close()
        except ImportError:
            logger.warning("openpyxl not installed for Excel. Run: pip install openpyxl")
            data = [{"error": "openpyxl not installed"}]

    elif ext == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore
            table = pq.read_table(path)
            df_dict = table.to_pydict()
            keys = list(df_dict.keys())
            n = len(df_dict[keys[0]]) if keys else 0
            for i in range(min(n, max_samples or n)):
                data.append({k: df_dict[k][i] for k in keys})
        except ImportError:
            logger.warning("pyarrow not installed for Parquet. Run: pip install pyarrow")
            data = [{"error": "pyarrow not installed"}]

    elif ext == ".txt":
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    data.append({"text": line})
                    if max_samples and len(data) >= max_samples:
                        break

    elif ext == ".pdf":
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text and text.strip():
                        data.append({"text": text.strip()})
                        if max_samples and len(data) >= max_samples:
                            break
        except ImportError:
            logger.warning("pdfplumber not installed for PDF. Run: pip install pdfplumber")
            data = [{"error": "pdfplumber not installed"}]

    else:
        # Fallback: try as text
        with open(path, encoding="utf-8", errors="replace") as f:
            data = [{"text": f.read()}]

    if max_samples and len(data) > max_samples:
        data = data[:max_samples]

    return data


def create_connector(source_type: str, **kwargs: Any) -> DataConnector:
    """Factory to create a connector by type name."""
    connectors = {
        "huggingface": HuggingFaceConnector,
        "s3": S3Connector,
        "gcs": GCSConnector,
        "azure": AzureBlobConnector,
        "google_drive": GoogleDriveConnector,
        "database": DatabaseConnector,
        "local": LocalFileConnector,
    }
    cls = connectors.get(source_type)
    if not cls:
        raise ValueError(f"Unknown source type: {source_type}. Available: {list(connectors.keys())}")
    return cls(**kwargs)
