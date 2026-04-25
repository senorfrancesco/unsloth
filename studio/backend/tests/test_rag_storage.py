# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.rag.storage import SQLiteRAGRepository


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii = False, sort_keys = True)


def test_repository_migrates_embedded_projection_to_projection_table(tmp_path: Path):
    db_path = tmp_path / "rag-migrate.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE rag_ingestion_profiles (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE rag_collections (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO rag_ingestion_profiles (id, payload_json)
            VALUES (?, ?)
            """,
            (
                "ing_legacy",
                _dump(
                    {
                        "id": "ing_legacy",
                        "name": "Legacy Profile",
                        "extractor": "docling",
                        "ocr_engine": "rapidocr",
                        "embedder": "bge-m3",
                        "reranker": None,
                        "chunk_size": 600,
                        "chunk_overlap": 80,
                        "created_at": "2026-04-22T00:00:00+00:00",
                    }
                ),
            ),
        )
        conn.execute(
            """
            INSERT INTO rag_collections (id, payload_json)
            VALUES (?, ?)
            """,
            (
                "ragcol_legacy",
                _dump(
                    {
                        "id": "ragcol_legacy",
                        "name": "Legacy Knowledge",
                        "backend": "qdrant",
                        "connection_profile_id": "conn_legacy",
                        "ingestion_profile_id": "ing_legacy",
                        "remote_collection_name": "legacy_knowledge",
                        "documents_count": 3,
                        "chunks_count": 14,
                        "sync_status": "pending",
                        "last_job_status": None,
                        "last_error": None,
                        "last_reindex_at": None,
                        "active_projection": {
                            "id": "ragcol_legacy:qdrant:v1",
                            "backend": "qdrant",
                            "embedder": "bge-m3",
                            "chunk_recipe": "600/80",
                            "status": "ready",
                            "version": 1,
                            "indexed_at": "2026-04-22T00:00:00+00:00",
                        },
                        "created_at": "2026-04-22T00:00:00+00:00",
                    }
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    repository = SQLiteRAGRepository(db_path)
    migrated_collection = repository.get_collection("ragcol_legacy")

    assert migrated_collection is not None
    assert "active_projection" not in migrated_collection
    assert migrated_collection["active_projection_id"] == "ragcol_legacy:qdrant:v1"

    projection = repository.get_projection("ragcol_legacy:qdrant:v1")
    assert projection is not None
    assert projection["physical_collection_name"] == "legacy_knowledge__proj_v1"
    assert projection["extractor"] == "docling"
    assert projection["ocr_engine"] == "rapidocr"
    assert projection["embedding_model"] == "bge-m3"
    assert projection["chunk_recipe"] == "char:600:80"
    assert projection["source_document_count"] == 3


def test_repository_persists_rag_module_records(tmp_path: Path):
    repository = SQLiteRAGRepository(tmp_path / "rag-modules.db")
    installation = {
        "module_id": "docling",
        "kind": "extractor",
        "status": "missing",
        "source_type": "python_package",
        "version": None,
        "path": None,
        "last_checked_at": "2026-04-22T00:00:00+00:00",
        "last_error": "not installed",
        "install_command": "python -m pip install docling",
    }
    instance = {
        "id": "ragmod_local",
        "name": "Local BGE-M3",
        "module_id": "bge-m3",
        "kind": "embedder",
        "source_type": "local_model_path",
        "model_id": None,
        "local_path": "/models/bge-m3",
        "service_url": None,
        "binary_path": None,
        "enabled": True,
        "status": "configured",
        "last_checked_at": None,
        "last_error": None,
        "created_at": "2026-04-22T00:00:00+00:00",
    }

    repository.upsert_module_installation(installation)
    repository.upsert_module_instance(instance)

    assert repository.get_module_installation("docling") == installation
    assert repository.get_module_instance("ragmod_local") == instance
    assert repository.list_module_installations() == [installation]
    assert repository.list_module_instances() == [instance]
    assert repository.delete_module_instance("ragmod_local") is True
    assert repository.delete_module_instance("ragmod_local") is False
    assert repository.get_module_instance("ragmod_local") is None
