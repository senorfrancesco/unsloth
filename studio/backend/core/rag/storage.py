# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from utils.paths import ensure_dir, studio_root


class SQLiteRAGRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        ensure_dir(self._db_path.parent)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    self._ensure_schema(conn)
                    self._schema_ready = True
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_connection_profiles (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_ingestion_profiles (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_module_installations (
                module_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_module_instances (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_datasets (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_dataset_documents (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_dataset_documents_dataset_id ON rag_dataset_documents(dataset_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_collections (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_index_projections (
                id TEXT PRIMARY KEY,
                collection_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE (collection_id, version)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_rag_index_projections_collection_id
            ON rag_index_projections(collection_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_collection_datasets (
                collection_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                PRIMARY KEY (collection_id, dataset_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_jobs (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL
            )
            """
        )
        SQLiteRAGRepository._migrate_embedded_projections(conn)

    @staticmethod
    def _dump(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii = False, sort_keys = True)

    @staticmethod
    def _load(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return json.loads(row["payload_json"])

    @staticmethod
    def _load_many(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
        return [json.loads(row["payload_json"]) for row in rows]

    @staticmethod
    def _chunk_recipe(chunk_size: int, chunk_overlap: int) -> str:
        return f"char:{chunk_size}:{chunk_overlap}"

    @staticmethod
    def _normalize_chunk_recipe(recipe: str, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> str:
        value = recipe.strip()
        if value.startswith("char:"):
            return value
        match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
        if match:
            return SQLiteRAGRepository._chunk_recipe(
                int(match.group(1)),
                int(match.group(2)),
            )
        if chunk_size is not None and chunk_overlap is not None:
            return SQLiteRAGRepository._chunk_recipe(chunk_size, chunk_overlap)
        return value or SQLiteRAGRepository._chunk_recipe(900, 120)

    @staticmethod
    def _physical_collection_name(remote_collection_name: str, version: int) -> str:
        return f"{remote_collection_name}__proj_v{version}"

    @staticmethod
    def _get_payload_by_id(
        conn: sqlite3.Connection,
        table_name: str,
        record_id: str | None,
    ) -> dict[str, Any] | None:
        if not record_id:
            return None
        row = conn.execute(
            f"SELECT payload_json FROM {table_name} WHERE id = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    @staticmethod
    def _migrate_embedded_projections(conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT id, payload_json FROM rag_collections ORDER BY id"
        ).fetchall()
        migrated = False
        for row in rows:
            payload = json.loads(row["payload_json"])
            embedded_projection = payload.get("active_projection")
            active_projection_id = payload.get("active_projection_id")
            if embedded_projection is None and active_projection_id:
                continue
            if embedded_projection is None:
                continue

            version = int(embedded_projection.get("version", 1))
            projection_id = str(
                embedded_projection.get("id")
                or active_projection_id
                or f"{payload['id']}:{payload.get('backend', 'qdrant')}:v{version}"
            )
            ingestion = SQLiteRAGRepository._get_payload_by_id(
                conn,
                "rag_ingestion_profiles",
                payload.get("ingestion_profile_id"),
            ) or {}
            chunk_size = int(ingestion.get("chunk_size", 900))
            chunk_overlap = int(ingestion.get("chunk_overlap", 120))
            embedder = str(
                embedded_projection.get("embedding_model")
                or embedded_projection.get("embedder")
                or ingestion.get("embedder")
                or "unknown"
            )
            migrated_projection = {
                "id": projection_id,
                "collection_id": payload["id"],
                "backend": embedded_projection.get("backend", payload.get("backend", "qdrant")),
                "embedder": embedder,
                "embedding_model": embedder,
                "physical_collection_name": str(
                    embedded_projection.get("physical_collection_name")
                    or SQLiteRAGRepository._physical_collection_name(
                        str(payload.get("remote_collection_name", "rag_collection")),
                        version,
                    )
                ),
                "extractor": str(
                    embedded_projection.get("extractor")
                    or ingestion.get("extractor")
                    or "unknown"
                ),
                "ocr_engine": str(
                    embedded_projection.get("ocr_engine")
                    or ingestion.get("ocr_engine")
                    or "unknown"
                ),
                "chunk_recipe": SQLiteRAGRepository._normalize_chunk_recipe(
                    str(embedded_projection.get("chunk_recipe", "")),
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                ),
                "status": str(embedded_projection.get("status", "pending")),
                "version": version,
                "source_document_count": int(
                    embedded_projection.get("source_document_count")
                    or payload.get("documents_count")
                    or 0
                ),
                "indexed_at": embedded_projection.get("indexed_at"),
            }
            conn.execute(
                """
                INSERT INTO rag_index_projections (id, collection_id, version, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection_id = excluded.collection_id,
                    version = excluded.version,
                    payload_json = excluded.payload_json
                """,
                (
                    projection_id,
                    payload["id"],
                    version,
                    SQLiteRAGRepository._dump(migrated_projection),
                ),
            )
            payload.pop("active_projection", None)
            payload["active_projection_id"] = projection_id
            conn.execute(
                """
                UPDATE rag_collections
                SET payload_json = ?
                WHERE id = ?
                """,
                (SQLiteRAGRepository._dump(payload), payload["id"]),
            )
            migrated = True
        if migrated:
            conn.commit()

    def list_connection_profiles(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM rag_connection_profiles ORDER BY id"
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_connection_profile(self, profile_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_connection_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_connection_profile(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_connection_profiles (id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_ingestion_profiles(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM rag_ingestion_profiles ORDER BY id"
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_ingestion_profile(self, profile_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_ingestion_profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_ingestion_profile(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_ingestion_profiles (id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_module_installations(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM rag_module_installations ORDER BY module_id"
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_module_installation(self, module_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_module_installations WHERE module_id = ?",
                (module_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_module_installation(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_module_installations (module_id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(module_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["module_id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_module_instances(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT payload_json FROM rag_module_instances ORDER BY id"
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_module_instance(self, instance_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_module_instances WHERE id = ?",
                (instance_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_module_instance(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_module_instances (id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_module_instance(self, instance_id: str) -> bool:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM rag_module_instances WHERE id = ?",
                (instance_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_datasets(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT payload_json FROM rag_datasets ORDER BY id").fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_datasets WHERE id = ?",
                (dataset_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_dataset(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_datasets (id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_dataset_documents(self, dataset_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM rag_dataset_documents
                WHERE dataset_id = ?
                ORDER BY id
                """,
                (dataset_id,),
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def insert_dataset_document(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_dataset_documents (id, dataset_id, payload_json)
                VALUES (?, ?, ?)
                """,
                (payload["id"], payload["dataset_id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_collections(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT payload_json FROM rag_collections ORDER BY id").fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_collections WHERE id = ?",
                (collection_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_collection(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_collections (id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()

    def list_collection_projections(self, collection_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM rag_index_projections
                WHERE collection_id = ?
                ORDER BY version
                """,
                (collection_id,),
            ).fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def get_projection(self, projection_id: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM rag_index_projections WHERE id = ?",
                (projection_id,),
            ).fetchone()
            return self._load(row)
        finally:
            conn.close()

    def upsert_projection(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO rag_index_projections (id, collection_id, version, payload_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    collection_id = excluded.collection_id,
                    version = excluded.version,
                    payload_json = excluded.payload_json
                """,
                (
                    payload["id"],
                    payload["collection_id"],
                    payload["version"],
                    self._dump(payload),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def link_dataset_to_collection(self, collection_id: str, dataset_id: str) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO rag_collection_datasets (collection_id, dataset_id)
                VALUES (?, ?)
                """,
                (collection_id, dataset_id),
            )
            conn.commit()
        finally:
            conn.close()

    def list_collection_dataset_ids(self, collection_id: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT dataset_id
                FROM rag_collection_datasets
                WHERE collection_id = ?
                ORDER BY dataset_id
                """,
                (collection_id,),
            ).fetchall()
            return [str(row["dataset_id"]) for row in rows]
        finally:
            conn.close()

    def list_jobs(self) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT payload_json FROM rag_jobs ORDER BY id").fetchall()
            return self._load_many(rows)
        finally:
            conn.close()

    def insert_job(self, payload: dict[str, Any]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO rag_jobs (id, payload_json) VALUES (?, ?)",
                (payload["id"], self._dump(payload)),
            )
            conn.commit()
        finally:
            conn.close()


def _sqlite_path_from_dsn(dsn: str) -> Path:
    raw = dsn.removeprefix("sqlite:///")
    if not raw:
        raise RuntimeError("Invalid `RAG_DB_DSN`: sqlite path is empty.")
    return Path(raw).expanduser()


def create_rag_repository_from_env() -> SQLiteRAGRepository:
    dsn = os.getenv("RAG_DB_DSN", "").strip()
    if dsn:
        if dsn.startswith("sqlite:///"):
            return SQLiteRAGRepository(_sqlite_path_from_dsn(dsn))
        if dsn.startswith("postgresql://") or dsn.startswith("postgres://"):
            raise RuntimeError(
                "`RAG_DB_DSN` points to PostgreSQL, but the current runtime does not "
                "include a PostgreSQL `RAG` repository implementation yet."
            )
        raise RuntimeError("Unsupported `RAG_DB_DSN`. Use `sqlite:///...`.")
    return SQLiteRAGRepository(studio_root() / "rag.db")
