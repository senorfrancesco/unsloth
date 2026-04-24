# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import importlib.util
import os
import sys
import types
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

from core.rag.service import RAGService, reset_rag_service
from core.rag.storage import SQLiteRAGRepository

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

    def health_check(self) -> dict[str, object]:
        return {
            "status": "healthy",
            "details": {"backend": "qdrant", "transport": "fake"},
        }

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


def test_modules_catalog_checks_instances_and_ingestion_overrides(tmp_path: Path):
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
