# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import gzip
import importlib.util
import io
import os
import sys
import types
import zipfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import core.rag.service as rag_service_module
from core.rag.service import RAGService, reset_rag_service
from core.rag.storage import SQLiteRAGRepository
from models.rag import RAGModuleInstallationSummary

_rag_route_path = Path(__file__).resolve().parent.parent / "routes" / "rag.py"
_rag_spec = importlib.util.spec_from_file_location("test_routes_rag", _rag_route_path)
assert _rag_spec is not None and _rag_spec.loader is not None
_rag_module = importlib.util.module_from_spec(_rag_spec)
_rag_spec.loader.exec_module(_rag_module)
router = _rag_module.router


class FakeEmbeddingProvider:
    engine = "fake-embeddings"

    def embed_texts(self, model_id: str, texts: list[str]) -> list[list[float]]:
        base = float(len(model_id))
        return [[base, float(len(text)), float(index + 1)] for index, text in enumerate(texts)]


class FakeVectorStoreAdapter:
    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, object]]] = {}
        self.health_status = "healthy"

    def health_check(self) -> dict[str, object]:
        return {
            "status": self.health_status,
            "details": {"backend": "qdrant", "transport": "fake"},
        }

    def get_collection_info(self, *, collection_name: str) -> dict[str, object]:
        if collection_name not in self.collections:
            return {
                "status": "missing",
                "details": {"collection_name": collection_name},
            }
        return {
            "status": "found",
            "details": {
                "status": "green",
                "points_count": len(self.collections[collection_name]),
            },
        }

    def scroll_points(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: object | None = None,
    ) -> dict[str, object]:
        if self.health_status != "healthy":
            raise RuntimeError("Qdrant is unavailable")
        items = self.collections.get(collection_name, [])
        start = offset if isinstance(offset, int) else 0
        page = items[start : start + limit]
        next_offset = start + len(page) if start + len(page) < len(items) else None
        return {
            "points": [
                {"id": point["id"], "payload": point["payload"]}
                for point in page
            ],
            "next_page_offset": next_offset,
        }

    def search_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
    ) -> dict[str, object]:
        if self.health_status != "healthy":
            raise RuntimeError("Qdrant is unavailable")
        scored: list[dict[str, object]] = []
        for point in self.collections.get(collection_name, []):
            stored_vector = point.get("vector")
            score = 0.0
            if isinstance(stored_vector, list):
                score = sum(float(left) * float(right) for left, right in zip(vector, stored_vector))
            scored.append(
                {
                    "id": point["id"],
                    "score": score,
                    "payload": point["payload"],
                }
            )
        scored.sort(key = lambda item: float(item["score"]), reverse = True)
        return {"points": scored[:limit]}

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        self.collections.setdefault(collection_name, [])

    def upsert_chunks(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]:
        bucket = self.collections.setdefault(collection_name, [])
        bucket.extend(points)
        return {"upserted_points": len(points)}

    def replace_projection(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]:
        self.collections[collection_name] = list(points)
        return {"upserted_points": len(points)}


def create_client(tmp_path: Path) -> TestClient:
    repository = SQLiteRAGRepository(tmp_path / "rag-test.db")
    adapter = FakeVectorStoreAdapter()
    service = RAGService(
        repository = repository,
        embedding_provider = FakeEmbeddingProvider(),
        vector_store_factory = lambda _profile: adapter,
        asset_root = tmp_path / "rag-assets",
    )
    reset_rag_service(service)
    app = FastAPI()
    app.include_router(router, prefix = "/api/rag", tags = ["rag"])
    app.state.rag_fake_adapter = adapter
    app.state.rag_service = service
    return TestClient(app)


def _create_connection_profile(client: TestClient) -> dict:
    response = client.post(
        "/api/rag/connection-profiles",
        json = {
            "name": "Primary Qdrant",
            "backend": "qdrant",
            "base_url": "http://qdrant.internal:6333",
            "api_key": "secret-token",
            "default_collection_prefix": "unsloth",
        },
    )
    assert response.status_code == 201
    return response.json()


def _create_ingestion_profile(client: TestClient) -> dict:
    response = client.post(
        "/api/rag/ingestion-profiles",
        json = {
            "name": "Docs Default",
            "extractor": "docling",
            "ocr_engine": "rapidocr",
            "embedder": "bge-m3",
            "reranker": "none",
            "chunk_size": 600,
            "chunk_overlap": 80,
        },
    )
    assert response.status_code == 201
    return response.json()


def _create_collection(client: TestClient, connection_profile: dict, ingestion_profile: dict) -> dict:
    response = client.post(
        "/api/rag/collections",
        json = {
            "name": "Support Knowledge",
            "connection_profile_id": connection_profile["id"],
            "ingestion_profile_id": ingestion_profile["id"],
            "remote_collection_name": "support_knowledge",
        },
    )
    assert response.status_code == 201
    return response.json()


def test_overview_and_profiles_are_exposed(tmp_path: Path):
    client = create_client(tmp_path)

    overview = client.get("/api/rag/overview")
    assert overview.status_code == 200
    assert overview.json()["collections_total"] == 0

    providers = client.get("/api/rag/providers")
    assert providers.status_code == 200
    assert providers.json()["vector_stores"][0]["id"] == "qdrant"

    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)

    connections = client.get("/api/rag/connection-profiles")
    assert connections.status_code == 200
    assert connections.json()["items"][0]["id"] == connection["id"]

    ingestion_profiles = client.get("/api/rag/ingestion-profiles")
    assert ingestion_profiles.status_code == 200
    assert ingestion_profiles.json()["items"][0]["id"] == ingestion["id"]


def test_modules_catalog_checks_instances_and_ingestion_overrides(tmp_path: Path, monkeypatch):
    client = create_client(tmp_path)

    catalog_response = client.get("/api/rag/modules/catalog")
    assert catalog_response.status_code == 200
    catalog_ids = {item["id"] for item in catalog_response.json()["items"]}
    assert {"docling", "bge-m3", "qdrant"}.issubset(catalog_ids)

    check_response = client.post("/api/rag/modules/docling/check")
    assert check_response.status_code == 200
    assert check_response.json()["status"] in {"available", "missing", "error"}

    install_response = client.post("/api/rag/modules/qdrant/install")
    assert install_response.status_code == 400

    model_dir = tmp_path / "models" / "bge-m3"
    model_dir.mkdir(parents = True)
    instance_response = client.post(
        "/api/rag/modules/instances",
        json = {
            "name": "Local BGE-M3",
            "module_id": "bge-m3",
            "source_type": "local_model_path",
            "local_path": str(model_dir),
            "enabled": True,
        },
    )
    assert instance_response.status_code == 201
    module_instance = instance_response.json()
    assert module_instance["kind"] == "embedder"
    assert module_instance["status"] == "configured"

    test_response = client.post(f"/api/rag/modules/instances/{module_instance['id']}/test")
    assert test_response.status_code == 200
    assert test_response.json()["status"] == "configured"

    connection = _create_connection_profile(client)
    ingestion_response = client.post(
        "/api/rag/ingestion-profiles",
        json = {
            "name": "Local Embedder Profile",
            "extractor": "docling",
            "ocr_engine": "rapidocr",
            "embedder": "bge-m3",
            "reranker": "none",
            "ocr_enabled": False,
            "reranker_enabled": False,
            "embedder_instance_id": module_instance["id"],
            "chunk_size": 600,
            "chunk_overlap": 80,
        },
    )
    assert ingestion_response.status_code == 201
    ingestion = ingestion_response.json()
    assert ingestion["ocr_engine"] == "none"
    assert ingestion["embedder_instance_id"] == module_instance["id"]

    collection = _create_collection(client, connection, ingestion)
    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Module Bound Dataset",
            "source_kind": "normalized-text",
        },
    ).json()
    append_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/text",
        json = {
            "document_name": "module.txt",
            "text": "Module aware embedding profile. " * 90,
        },
    )
    assert append_response.status_code == 201

    publish_response = client.post(
        f"/api/rag/collections/{collection['id']}/publish",
        json = {"dataset_id": dataset["id"]},
    )
    assert publish_response.status_code == 201

    collection_after_publish = client.get("/api/rag/collections").json()["items"][0]
    projection = collection_after_publish["active_projection"]
    assert projection["embedder"] == "bge-m3"
    assert projection["embedding_model"] == str(model_dir)
    assert projection["ocr_engine"] == "none"

    adapter = client.app.state.rag_fake_adapter
    first_payload = adapter.collections["support_knowledge__proj_v1"][0]["payload"]
    assert first_payload["embedding_config"] == {
        "engine": "fake-embeddings",
        "model": str(model_dir),
    }
    assert first_payload["ocr_engine"] == "none"

    delete_response = client.delete(f"/api/rag/modules/instances/{module_instance['id']}")
    assert delete_response.status_code == 200
    assert delete_response.json()["details"]["deleted_files"] is False
    assert delete_response.json()["details"]["cleared_ingestion_profile_ids"] == [ingestion["id"]]
    assert client.get("/api/rag/modules/instances").json()["items"] == []
    refreshed_ingestion = client.get("/api/rag/ingestion-profiles").json()["items"][0]
    assert refreshed_ingestion["embedder_instance_id"] is None

    def fake_uninstall(item):
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "missing",
            last_checked_at = "2026-04-24T00:00:00+00:00",
            last_error = None,
        )

    monkeypatch.setattr(rag_service_module, "uninstall_catalog_module", fake_uninstall)
    uninstall_response = client.delete("/api/rag/modules/docling/package")
    assert uninstall_response.status_code == 200
    assert uninstall_response.json()["status"] == "missing"


def test_text_dataset_publish_updates_collection_and_jobs(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)

    dataset_response = client.post(
        "/api/rag/datasets",
        json = {
            "name": "April Product Docs",
            "source_kind": "normalized-text",
            "description": "Normalized release notes and operator docs.",
        },
    )
    assert dataset_response.status_code == 201
    dataset = dataset_response.json()

    append_text_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/text",
        json = {
            "document_name": "release-notes.md",
            "text": "Alpha beta gamma. " * 120,
        },
    )
    assert append_text_response.status_code == 201

    publish_response = client.post(
        f"/api/rag/collections/{collection['id']}/publish",
        json = {"dataset_id": dataset["id"]},
    )
    assert publish_response.status_code == 201
    publish_job = publish_response.json()
    assert publish_job["job_type"] == "publish"
    assert publish_job["status"] == "completed"

    collections_response = client.get("/api/rag/collections")
    assert collections_response.status_code == 200
    updated_collection = collections_response.json()["items"][0]
    assert updated_collection["documents_count"] == 1
    assert updated_collection["chunks_count"] > 0
    assert updated_collection["last_job_status"] == "completed"

    datasets_response = client.get("/api/rag/datasets")
    assert datasets_response.status_code == 200
    updated_dataset = datasets_response.json()["items"][0]
    assert updated_dataset["documents_count"] == 1
    assert updated_dataset["chunks_count"] > 0

    jobs_response = client.get("/api/rag/jobs")
    assert jobs_response.status_code == 200
    assert jobs_response.json()["items"][0]["id"] == publish_job["id"]


def test_collection_inspector_handles_empty_collection(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)

    inspect_response = client.get(f"/api/rag/collections/{collection['id']}/inspect")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["collection"]["id"] == collection["id"]
    assert inspect_payload["active_projection"]["status"] == "pending"
    assert inspect_payload["qdrant"]["status"] == "missing"
    assert inspect_payload["stats"]["chunks_total"] == 0
    assert inspect_payload["warnings"]

    sample_response = client.get(f"/api/rag/collections/{collection['id']}/sample-chunks")
    assert sample_response.status_code == 200
    assert sample_response.json()["items"] == []


def test_collection_inspector_samples_and_searches_published_chunks(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)

    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Searchable Docs",
            "source_kind": "normalized-text",
        },
    ).json()
    append_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/text",
        json = {
            "document_name": "retrieval.md",
            "text": "Alpha beta gamma. Retrieval should find this document. " * 90,
        },
    )
    assert append_response.status_code == 201

    publish_response = client.post(
        f"/api/rag/collections/{collection['id']}/publish",
        json = {"dataset_id": dataset["id"]},
    )
    assert publish_response.status_code == 201

    inspect_response = client.get(f"/api/rag/collections/{collection['id']}/inspect")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["qdrant"]["status"] == "healthy"
    assert inspect_payload["stats"]["chunks_total"] > 0
    assert inspect_payload["stats"]["documents_total"] == 1
    assert inspect_payload["distributions"]["documents"][0]["label"] == "retrieval.md"
    assert inspect_payload["distributions"]["chunk_sizes"]
    assert inspect_payload["distributions"]["indexing_statuses"][0]["label"] == "indexed"

    sample_response = client.get(
        f"/api/rag/collections/{collection['id']}/sample-chunks",
        params = {"limit": 2, "offset": 0},
    )
    assert sample_response.status_code == 200
    sample_payload = sample_response.json()
    assert len(sample_payload["items"]) == 2
    sample_chunk = sample_payload["items"][0]
    assert "vector" not in sample_chunk
    assert sample_chunk["text"]
    assert sample_chunk["file_id"]
    assert sample_chunk["document_id"]
    assert sample_chunk["source"] == "retrieval.md"
    assert sample_chunk["indexed_at"]
    assert sample_chunk["embedding_config"] == {
        "engine": "fake-embeddings",
        "model": "bge-m3",
    }

    search_response = client.post(
        f"/api/rag/collections/{collection['id']}/search",
        json = {"query": "gamma retrieval", "limit": 3},
    )
    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert search_payload["embedding_model"] == "bge-m3"
    assert len(search_payload["results"]) == 3
    assert search_payload["results"][0]["score"] >= search_payload["results"][-1]["score"]
    assert "Retrieval should find" in search_payload["results"][0]["text"]


def test_collection_inspector_reports_unhealthy_qdrant(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)
    adapter = client.app.state.rag_fake_adapter
    adapter.health_status = "error"

    inspect_response = client.get(f"/api/rag/collections/{collection['id']}/inspect")
    assert inspect_response.status_code == 200
    inspect_payload = inspect_response.json()
    assert inspect_payload["qdrant"]["status"] == "error"
    assert "not healthy" in inspect_payload["warnings"][0]

    sample_response = client.get(f"/api/rag/collections/{collection['id']}/sample-chunks")
    assert sample_response.status_code == 503
    assert "Failed to read Qdrant collection chunks" in sample_response.json()["detail"]


def test_collection_inspector_rejects_unsupported_backend(tmp_path: Path):
    client = create_client(tmp_path)
    connection_response = client.post(
        "/api/rag/connection-profiles",
        json = {
            "name": "Postgres Preview",
            "backend": "pgvector",
            "base_url": "postgresql://pgvector.internal/rag",
        },
    )
    assert connection_response.status_code == 201
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection_response.json(), ingestion)

    inspect_response = client.get(f"/api/rag/collections/{collection['id']}/inspect")
    assert inspect_response.status_code == 400
    assert "Qdrant" in inspect_response.json()["detail"]


def test_publish_uses_projection_metadata_and_reindex_creates_new_projection(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)

    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Operator Runbooks",
            "source_kind": "documents",
        },
    ).json()

    append_text_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/text",
        json = {
            "document_name": "runbook.txt",
            "text": "First line.\nSecond line.\n" * 80,
        },
    )
    assert append_text_response.status_code == 201

    publish_response = client.post(
        f"/api/rag/collections/{collection['id']}/publish",
        json = {"dataset_id": dataset["id"]},
    )
    assert publish_response.status_code == 201

    collection_after_publish = client.get("/api/rag/collections").json()["items"][0]
    projection = collection_after_publish["active_projection"]
    assert projection["version"] == 1
    assert projection["physical_collection_name"] == "support_knowledge__proj_v1"
    assert projection["embedding_model"] == "bge-m3"
    assert projection["extractor"] == "docling"
    assert projection["ocr_engine"] == "rapidocr"
    assert projection["chunk_recipe"] == "char:600:80"
    assert projection["source_document_count"] == 1

    adapter = client.app.state.rag_fake_adapter
    assert "support_knowledge__proj_v1" in adapter.collections
    first_payload = adapter.collections["support_knowledge__proj_v1"][0]["payload"]
    assert first_payload["knowledge_base_id"] == collection["id"]
    assert first_payload["file_id"]
    assert first_payload["hash"]
    assert first_payload["extractor"] == "docling"
    assert first_payload["ocr_engine"] == "rapidocr"
    assert first_payload["created_by"] is None
    assert first_payload["chunk_recipe"] == "char:600:80"
    assert first_payload["embedding_config"] == {
        "engine": "fake-embeddings",
        "model": "bge-m3",
    }
    assert first_payload["indexed_at"]
    assert isinstance(first_payload["start_index"], int)
    assert first_payload["page"] is None
    assert first_payload["language"] is None

    reindex_response = client.post(f"/api/rag/collections/{collection['id']}/reindex")
    assert reindex_response.status_code == 201
    assert reindex_response.json()["job_type"] == "reindex"

    collection_after_reindex = client.get("/api/rag/collections").json()["items"][0]
    reindexed_projection = collection_after_reindex["active_projection"]
    assert reindexed_projection["version"] == 2
    assert reindexed_projection["physical_collection_name"] == "support_knowledge__proj_v2"
    assert "support_knowledge__proj_v1" in adapter.collections
    assert "support_knowledge__proj_v2" in adapter.collections


def test_reindex_uses_extracted_text_and_reports_missing_artifact(tmp_path: Path):
    client = create_client(tmp_path)
    connection = _create_connection_profile(client)
    ingestion = _create_ingestion_profile(client)
    collection = _create_collection(client, connection, ingestion)

    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Operator Runbooks",
            "source_kind": "documents",
        },
    ).json()

    upload_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/documents",
        files = {
            "file": ("runbook.txt", b"First line.\nSecond line.\n" * 80, "text/plain"),
        },
    )
    assert upload_response.status_code == 201
    assert upload_response.json()["status"] == "ok"
    dataset_detail = client.get(f"/api/rag/datasets/{dataset['id']}").json()
    uploaded_document = dataset_detail["documents"][0]
    assert uploaded_document["processing_status"] == "ready"
    assert uploaded_document["processing_error"] is None
    assert uploaded_document["processed_at"]
    assert uploaded_document["extractor"] == "builtin-file"

    publish_response = client.post(
        f"/api/rag/collections/{collection['id']}/publish",
        json = {"dataset_id": dataset["id"]},
    )
    assert publish_response.status_code == 201

    service = client.app.state.rag_service
    document = service._repository.list_dataset_documents(dataset["id"])[0]
    raw_path = Path(str(document["raw_path"]))
    extracted_path = Path(str(document["extracted_path"]))

    os.unlink(raw_path)

    reindex_response = client.post(f"/api/rag/collections/{collection['id']}/reindex")
    assert reindex_response.status_code == 201
    assert reindex_response.json()["job_type"] == "reindex"

    os.unlink(extracted_path)

    failed_reindex = client.post(f"/api/rag/collections/{collection['id']}/reindex")
    assert failed_reindex.status_code == 422
    assert "Missing extracted text artifact" in failed_reindex.json()["detail"]

    diagnostics = client.get("/api/rag/diagnostics")
    assert diagnostics.status_code == 200
    payload = diagnostics.json()
    assert payload["connection_profiles"][0]["status"] == "healthy"
    assert payload["collections"][0]["active_projection"]["version"] == 2
    assert payload["collections"][0]["last_job_status"] == "error"
    assert "Missing extracted text artifact" in payload["collections"][0]["last_error"]


def test_zip_upload_expands_supported_documents(tmp_path: Path):
    client = create_client(tmp_path)
    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Archive Uploads",
            "source_kind": "documents",
        },
    ).json()

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("docs/first.txt", "First document text.\n" * 8)
        archive.writestr("docs/second.md", "# Second\n\nSecond document text.\n" * 8)
        archive.writestr("images/ignored.png", b"not indexed")

    upload_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/documents",
        files = {
            "file": ("bundle.zip", archive_buffer.getvalue(), "application/zip"),
        },
    )
    assert upload_response.status_code == 201
    upload_payload = upload_response.json()
    assert upload_payload["status"] == "ok"
    assert len(upload_payload["files"]) == 2
    assert {item["filename"] for item in upload_payload["files"]} == {
        "docs/first.txt",
        "docs/second.md",
    }

    dataset_detail = client.get(f"/api/rag/datasets/{dataset['id']}").json()
    assert dataset_detail["dataset"]["documents_count"] == 2
    assert dataset_detail["dataset"]["status"] == "ready"
    assert {item["document_name"] for item in dataset_detail["documents"]} == {
        "docs/first.txt",
        "docs/second.md",
    }
    assert all(item["processing_status"] == "ready" for item in dataset_detail["documents"])


def test_gzip_upload_creates_inner_document(tmp_path: Path):
    client = create_client(tmp_path)
    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Compressed Uploads",
            "source_kind": "documents",
        },
    ).json()

    upload_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/documents",
        files = {
            "file": ("notes.txt.gz", gzip.compress(b"Compressed text.\n" * 10), "application/gzip"),
        },
    )
    assert upload_response.status_code == 201
    upload_payload = upload_response.json()
    assert len(upload_payload["files"]) == 1
    assert upload_payload["files"][0]["filename"] == "notes.txt"

    dataset_detail = client.get(f"/api/rag/datasets/{dataset['id']}").json()
    assert dataset_detail["dataset"]["documents_count"] == 1
    assert dataset_detail["documents"][0]["document_name"] == "notes.txt"


def test_file_upload_processing_error_is_visible_on_dataset(tmp_path: Path):
    client = create_client(tmp_path)
    dataset = client.post(
        "/api/rag/datasets",
        json = {
            "name": "Bad Uploads",
            "source_kind": "documents",
        },
    ).json()

    upload_response = client.post(
        f"/api/rag/datasets/{dataset['id']}/documents",
        files = {
            "file": ("archive.zip", b"not a document", "application/zip"),
        },
    )
    assert upload_response.status_code == 422
    assert "zip" in upload_response.json()["detail"]

    datasets_response = client.get("/api/rag/datasets")
    assert datasets_response.status_code == 200
    updated_dataset = datasets_response.json()["items"][0]
    assert updated_dataset["status"] == "error"
    assert "zip" in updated_dataset["last_error"]
