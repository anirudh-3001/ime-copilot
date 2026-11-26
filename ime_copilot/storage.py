"""Where original PDFs and derived files live.

  azure_blob - Azure Blob Storage (production). Use a private container with no
               public access, encryption at rest, and Entra ID (managed
               identity) instead of account keys.
  local      - a folder on disk, for development with synthetic data only.

Layout per case:
  cases/<case_id>/original.pdf   the file exactly as uploaded (never modified)
  cases/<case_id>/layout.json    Document Intelligence line polygons per page
  cases/<case_id>/ledger.json    the extracted ledger
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_case_id(case_id: str) -> str:
    if not _SAFE_ID.match(case_id):
        raise ValueError("case_id may contain only letters, digits, '-' and '_' (max 64).")
    return case_id


def case_key(case_id: str, name: str) -> str:
    return f"cases/{validate_case_id(case_id)}/{name}"


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Invalid storage key.")
        return path

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()


class AzureBlobStorage:
    def __init__(self, container: str, account_url: str = "", connection_string: str = "", client=None):
        from azure.storage.blob import BlobServiceClient  # pip install ime-copilot[azure]
        if client is not None:
            service = client
        elif account_url:
            from azure.identity import DefaultAzureCredential
            service = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        elif connection_string:
            service = BlobServiceClient.from_connection_string(connection_string)  # local dev only
        else:
            raise ValueError("Set AZURE_STORAGE_ACCOUNT_URL (preferred) or AZURE_STORAGE_CONNECTION_STRING.")
        self.container = service.get_container_client(container)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        from azure.storage.blob import ContentSettings
        blob = self.container.get_blob_client(key)
        blob.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type=content_type))
        return blob.url

    def get(self, key: str) -> bytes:
        return self.container.get_blob_client(key).download_blob().readall()

    def exists(self, key: str) -> bool:
        return self.container.get_blob_client(key).exists()


def get_storage(settings) -> Storage:
    if settings.storage == "azure_blob":
        return AzureBlobStorage(settings.azure_storage_container, settings.azure_storage_account_url,
                                settings.azure_storage_connection_string)
    return LocalStorage(settings.local_storage_dir)
