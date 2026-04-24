# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Callable
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from core.rag.documents import (
    build_qdrant_points,
    chunk_text,
    read_extracted_text,
    store_uploaded_document,
    write_text_document,
)
from core.rag.embedding import (
    EmbeddingProviderProtocol,
    SentenceTransformerEmbeddingProvider,
)
from core.rag.modules import (
    check_catalog_module,
    check_module_instance,
    install_catalog_module,
    install_local_catalog_module,
)
from core.rag.registry import get_rag_module_by_id, get_rag_module_catalog, get_rag_providers
from core.rag.storage import SQLiteRAGRepository, create_rag_repository_from_env
from core.rag.vector_store import (
    VectorStoreAdapterProtocol,
    build_vector_store_adapter,
)
from models.rag import (
    AppendRAGDatasetTextRequest,
    CreateRAGCollectionRequest,
    CreateRAGConnectionProfileRequest,
    CreateRAGDatasetRequest,
    CreateRAGIngestionProfileRequest,
    CreateRAGModuleInstanceRequest,
    InstallRAGModuleRequest,
    PublishRAGDatasetRequest,
    RAGCollectionListResponse,
    RAGCollectionSummary,
    RAGConnectionDiagnostic,
    RAGConnectionProfileListResponse,
    RAGConnectionProfileSummary,
    RAGDatasetDetailResponse,
    RAGDatasetDocumentSummary,
    RAGDatasetListResponse,
    RAGDatasetSummary,
    RAGDiagnosticsResponse,
    RAGDocumentUploadResponse,
    RAGIndexProjectionSummary,
    RAGIngestionProfileListResponse,
    RAGIngestionProfileSummary,
    RAGJobListResponse,
    RAGJobSummary,
    RAGModuleActionResponse,
    RAGModuleCatalogItem,
    RAGModuleCatalogResponse,
    RAGModuleInstallationListResponse,
    RAGModuleInstallationSummary,
    RAGModuleInstanceListResponse,
    RAGModuleInstanceSummary,
    RAGOverviewBackend,
    RAGOverviewResponse,
    RAGProvidersResponse,
)
from utils.paths import datasets_root, ensure_dir


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond = 0).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _clean_collection_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "rag_collection"


class RAGService:
    def __init__(
        self,
        *,
        repository: SQLiteRAGRepository | None = None,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        vector_store_factory: Callable[[dict], VectorStoreAdapterProtocol] | None = None,
        asset_root: Path | None = None,
    ) -> None:
        self._lock = RLock()
        self._repository = repository or create_rag_repository_from_env()
        self._embedding_provider = embedding_provider or SentenceTransformerEmbeddingProvider()
        self._vector_store_factory = vector_store_factory or build_vector_store_adapter
        self._asset_root = asset_root or ensure_dir(datasets_root() / "rag")
        ensure_dir(self._asset_root)

    def get_providers(self) -> RAGProvidersResponse:
        return get_rag_providers()

    def get_module_catalog(self) -> RAGModuleCatalogResponse:
        return get_rag_module_catalog()

    def list_module_installations(self) -> RAGModuleInstallationListResponse:
        stored = {
            item["module_id"]: item
            for item in self._repository.list_module_installations()
        }
        items: list[RAGModuleInstallationSummary] = []
        for catalog_item in get_rag_module_catalog().items:
            saved = stored.get(catalog_item.id)
            summary = (
                RAGModuleInstallationSummary.model_validate(saved)
                if saved is not None
                else check_catalog_module(catalog_item)
            )
            items.append(summary)
        items.sort(key = lambda item: (item.kind, item.module_id))
        return RAGModuleInstallationListResponse(items = items)

    def check_module(self, module_id: str) -> RAGModuleActionResponse:
        catalog_item = self._require_module(module_id)
        summary = check_catalog_module(catalog_item)
        self._repository.upsert_module_installation(summary.model_dump())
        return self._module_action_response(
            summary,
            message = "Module check completed.",
        )

    def install_module(self, module_id: str) -> RAGModuleActionResponse:
        catalog_item = self._require_module(module_id)
        if not catalog_item.installable:
            raise HTTPException(
                status_code = 400,
                detail = "This module is not allowlisted for managed online install.",
            )
        summary = install_catalog_module(catalog_item)
        self._repository.upsert_module_installation(summary.model_dump())
        return self._module_action_response(
            summary,
            message = "Managed module install completed.",
        )

    def install_local_module(
        self,
        module_id: str,
        payload: InstallRAGModuleRequest,
    ) -> RAGModuleActionResponse:
        catalog_item = self._require_module(module_id)
        summary = install_local_catalog_module(
            catalog_item,
            package_path = payload.package_path,
            wheelhouse_path = payload.wheelhouse_path,
        )
        self._repository.upsert_module_installation(summary.model_dump())
        return self._module_action_response(
            summary,
            message = "Local module install completed.",
        )

    def list_module_instances(self) -> RAGModuleInstanceListResponse:
        items = [
            RAGModuleInstanceSummary.model_validate(item)
            for item in self._repository.list_module_instances()
        ]
        items.sort(key = lambda item: (item.kind, item.name.lower()))
        return RAGModuleInstanceListResponse(items = items)

    def create_module_instance(
        self,
        payload: CreateRAGModuleInstanceRequest,
    ) -> RAGModuleInstanceSummary:
        catalog_item = self._require_module(payload.module_id)
        record = {
            "id": _new_id("ragmod"),
            "name": payload.name.strip(),
            "module_id": catalog_item.id,
            "kind": catalog_item.kind,
            "source_type": payload.source_type,
            "model_id": payload.model_id or catalog_item.default_model_id,
            "local_path": payload.local_path,
            "service_url": payload.service_url,
            "binary_path": payload.binary_path,
            "enabled": payload.enabled,
            "status": "configured",
            "last_checked_at": None,
            "last_error": None,
            "created_at": _now_iso(),
        }
        summary = check_module_instance(catalog_item, record)
        self._repository.upsert_module_instance(summary.model_dump())
        return summary

    def test_module_instance(self, instance_id: str) -> RAGModuleActionResponse:
        record = self._require_module_instance(instance_id)
        catalog_item = self._require_module(str(record["module_id"]))
        summary = check_module_instance(catalog_item, record)
        self._repository.upsert_module_instance(summary.model_dump())
        return RAGModuleActionResponse(
            module_id = summary.module_id,
            status = summary.status,
            message = "Module instance test completed.",
            details = summary.model_dump(),
        )

    def get_overview(self) -> RAGOverviewResponse:
        datasets = self._repository.list_datasets()
        collections = self._repository.list_collections()
        jobs = self._repository.list_jobs()
        return RAGOverviewResponse(
            collections_total = len(collections),
            jobs_total = len(jobs),
            datasets_total = len(datasets),
            documents_total = sum(int(item.get("documents_count", 0)) for item in datasets),
            chunks_total = sum(int(item.get("chunks_count", 0)) for item in datasets),
            synced_collections_total = sum(
                1 for item in collections if item.get("sync_status") == "synced"
            ),
            supported_backends = [
                RAGOverviewBackend(id = "qdrant", enabled = True, stage = "first-class"),
                RAGOverviewBackend(id = "pgvector", enabled = False, stage = "design-only"),
            ],
        )

    def list_connection_profiles(self) -> RAGConnectionProfileListResponse:
        items = [
            self._to_connection_summary(item)
            for item in self._repository.list_connection_profiles()
        ]
        items.sort(key = lambda item: item.name.lower())
        return RAGConnectionProfileListResponse(items = items)

    def create_connection_profile(
        self, payload: CreateRAGConnectionProfileRequest
    ) -> RAGConnectionProfileSummary:
        record = {
            "id": _new_id("ragconn"),
            "name": payload.name.strip(),
            "backend": payload.backend,
            "base_url": str(payload.base_url).rstrip("/"),
            "api_key": payload.api_key,
            "default_collection_prefix": payload.default_collection_prefix,
            "enabled": payload.enabled,
            "status": "unknown",
            "created_at": _now_iso(),
        }
        self._repository.upsert_connection_profile(record)
        return self._to_connection_summary(record)

    def list_ingestion_profiles(self) -> RAGIngestionProfileListResponse:
        items = [
            RAGIngestionProfileSummary.model_validate(item)
            for item in self._repository.list_ingestion_profiles()
        ]
        items.sort(key = lambda item: item.name.lower())
        return RAGIngestionProfileListResponse(items = items)

    def create_ingestion_profile(
        self, payload: CreateRAGIngestionProfileRequest
    ) -> RAGIngestionProfileSummary:
        self._validate_module_instance_kind(payload.extractor_instance_id, "extractor")
        self._validate_module_instance_kind(payload.ocr_instance_id, "ocr")
        self._validate_module_instance_kind(payload.embedder_instance_id, "embedder")
        self._validate_module_instance_kind(payload.reranker_instance_id, "reranker")
        record = {
            "id": _new_id("ragprof"),
            "name": payload.name.strip(),
            "extractor": payload.extractor,
            "ocr_engine": payload.ocr_engine if payload.ocr_enabled else "none",
            "embedder": payload.embedder,
            "reranker": None
            if not payload.reranker_enabled or payload.reranker in {None, "", "none"}
            else payload.reranker,
            "extractor_enabled": payload.extractor_enabled,
            "ocr_enabled": payload.ocr_enabled,
            "reranker_enabled": payload.reranker_enabled,
            "extractor_instance_id": payload.extractor_instance_id,
            "ocr_instance_id": payload.ocr_instance_id,
            "embedder_instance_id": payload.embedder_instance_id,
            "reranker_instance_id": payload.reranker_instance_id,
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
            "created_at": _now_iso(),
        }
        self._repository.upsert_ingestion_profile(record)
        return RAGIngestionProfileSummary.model_validate(record)

    def list_datasets(self) -> RAGDatasetListResponse:
        items = [
            RAGDatasetSummary.model_validate(item)
            for item in self._repository.list_datasets()
        ]
        items.sort(key = lambda item: item.updated_at, reverse = True)
        return RAGDatasetListResponse(items = items)

    def get_dataset_detail(self, dataset_id: str) -> RAGDatasetDetailResponse:
        dataset = self._require_dataset(dataset_id)
        documents = [
            RAGDatasetDocumentSummary.model_validate(item)
            for item in self._repository.list_dataset_documents(dataset_id)
        ]
        return RAGDatasetDetailResponse(
            dataset = RAGDatasetSummary.model_validate(dataset),
            documents = documents,
        )

    def create_dataset(self, payload: CreateRAGDatasetRequest) -> RAGDatasetSummary:
        now = _now_iso()
        record = {
            "id": _new_id("ragds"),
            "name": payload.name.strip(),
            "source_kind": payload.source_kind,
            "description": payload.description,
            "status": "empty",
            "documents_count": 0,
            "chunks_count": 0,
            "total_characters": 0,
            "last_error": None,
            "created_at": now,
            "updated_at": now,
        }
        self._repository.upsert_dataset(record)
        return RAGDatasetSummary.model_validate(record)

    def append_dataset_text(
        self, dataset_id: str, payload: AppendRAGDatasetTextRequest
    ) -> RAGDatasetDocumentSummary:
        with self._lock:
            dataset = self._require_dataset(dataset_id)
            dataset_dir = self._dataset_dir(dataset_id)
            stored = write_text_document(
                dataset_dir = dataset_dir,
                document_name = payload.document_name,
                text = payload.text,
            )
            document = {
                "id": stored["id"],
                "dataset_id": dataset_id,
                "document_name": stored["document_name"],
                "mime_type": stored["mime_type"],
                "source_kind": dataset["source_kind"],
                "size_bytes": stored["size_bytes"],
                "text_char_count": stored["text_char_count"],
                "content_hash": stored["content_hash"],
                "raw_path": stored["raw_path"],
                "extracted_path": stored["extracted_path"],
                "metadata_path": stored["metadata_path"],
                "created_at": _now_iso(),
            }
            self._repository.insert_dataset_document(document)
            self._refresh_dataset_stats(dataset_id)
            return RAGDatasetDocumentSummary.model_validate(document)

    async def upload_dataset_document(
        self, dataset_id: str, file: UploadFile
    ) -> RAGDocumentUploadResponse:
        with self._lock:
            dataset = self._require_dataset(dataset_id)
            dataset_dir = self._dataset_dir(dataset_id)
            stored = await store_uploaded_document(dataset_dir = dataset_dir, file = file)
            document = {
                "id": stored["id"],
                "dataset_id": dataset_id,
                "document_name": stored["document_name"],
                "mime_type": stored["mime_type"],
                "source_kind": dataset["source_kind"],
                "size_bytes": stored["size_bytes"],
                "text_char_count": stored["text_char_count"],
                "content_hash": stored["content_hash"],
                "raw_path": stored["raw_path"],
                "extracted_path": stored["extracted_path"],
                "metadata_path": stored["metadata_path"],
                "created_at": _now_iso(),
            }
            self._repository.insert_dataset_document(document)
            self._refresh_dataset_stats(dataset_id)
            return RAGDocumentUploadResponse(
                file_id = str(document["id"]),
                filename = str(document["document_name"]),
                size_bytes = int(document["size_bytes"]),
                status = "ok",
            )

    def list_collections(self) -> RAGCollectionListResponse:
        items = [self._to_collection_summary(item) for item in self._repository.list_collections()]
        items.sort(key = lambda item: item.name.lower())
        return RAGCollectionListResponse(items = items)

    def create_collection(self, payload: CreateRAGCollectionRequest) -> RAGCollectionSummary:
        connection = self._require_connection_profile(payload.connection_profile_id)
        ingestion = self._require_ingestion_profile(payload.ingestion_profile_id)
        now = _now_iso()
        collection_id = _new_id("ragcol")
        remote_collection_name = _clean_collection_name(payload.remote_collection_name)
        components = self._ingestion_components(ingestion)
        projection = self._build_projection(
            collection_id = collection_id,
            remote_collection_name = remote_collection_name,
            backend = connection["backend"],
            extractor = components["extractor"],
            ocr_engine = components["ocr_engine"],
            embedder = str(ingestion["embedder"]),
            embedding_model = components["embedding_model"],
            chunk_size = int(ingestion["chunk_size"]),
            chunk_overlap = int(ingestion["chunk_overlap"]),
            version = 1,
            status = "pending",
            indexed_at = None,
            source_document_count = 0,
        )
        record = {
            "id": collection_id,
            "name": payload.name.strip(),
            "backend": connection["backend"],
            "connection_profile_id": connection["id"],
            "ingestion_profile_id": ingestion["id"],
            "remote_collection_name": remote_collection_name,
            "documents_count": 0,
            "chunks_count": 0,
            "sync_status": "not-synced",
            "last_job_status": None,
            "last_error": None,
            "last_reindex_at": None,
            "active_projection_id": projection["id"],
            "created_at": now,
        }
        self._repository.upsert_projection(projection)
        self._repository.upsert_collection(record)
        return self._to_collection_summary(record)

    def publish_dataset(
        self,
        collection_id: str,
        payload: PublishRAGDatasetRequest,
        initiator: str = "admin",
    ) -> RAGJobSummary:
        with self._lock:
            collection = self._require_collection(collection_id)
            try:
                connection = self._require_connection_profile(collection["connection_profile_id"])
                ingestion = self._require_ingestion_profile(collection["ingestion_profile_id"])
                dataset = self._require_dataset(payload.dataset_id)
                documents = self._repository.list_dataset_documents(payload.dataset_id)
                if not documents:
                    raise HTTPException(status_code = 400, detail = "RAG dataset has no documents.")

                projection = self._require_projection(collection["active_projection_id"])
                chunk_texts = self._materialize_chunk_texts(documents, ingestion)
                components = self._ingestion_components(ingestion)
                vectors = self._embedding_provider.embed_texts(components["embedding_model"], chunk_texts)
                if not vectors:
                    raise HTTPException(status_code = 400, detail = "No chunks generated for publish.")

                adapter = self._vector_store_factory(connection)
                adapter.ensure_collection(
                    collection_name = str(projection["physical_collection_name"]),
                    vector_size = len(vectors[0]),
                )
                points, total_chunks = build_qdrant_points(
                    collection_id = collection_id,
                    projection_id = str(projection["id"]),
                    documents = documents,
                    chunk_size = int(ingestion["chunk_size"]),
                    chunk_overlap = int(ingestion["chunk_overlap"]),
                    extractor = components["extractor"],
                    ocr_engine = components["ocr_engine"],
                    chunk_recipe = str(projection["chunk_recipe"]),
                    embedding_config = self._embedding_config(components["embedding_model"]),
                    vectors = vectors,
                )
                adapter.upsert_chunks(
                    collection_name = str(projection["physical_collection_name"]),
                    points = points,
                )

                self._repository.link_dataset_to_collection(collection_id, payload.dataset_id)
                self._refresh_dataset_stats(payload.dataset_id, chunk_count = total_chunks)
                self._refresh_collection_stats(collection_id)

                collection = self._require_collection(collection_id)
                projection["status"] = "ready"
                projection["indexed_at"] = _now_iso()
                projection["source_document_count"] = int(collection["documents_count"])
                self._repository.upsert_projection(projection)
                collection["last_job_status"] = "completed"
                collection["sync_status"] = "pending"
                collection["last_error"] = None
                self._repository.upsert_collection(collection)

                job = self._register_job(
                    job_type = "publish",
                    collection = collection,
                    initiator = initiator,
                    dataset = dataset,
                    stage_label = "Published dataset to vector store",
                )
                return RAGJobSummary.model_validate(job)
            except HTTPException as exc:
                self._mark_collection_error(collection, detail = exc.detail)
                raise
            except Exception as exc:
                self._mark_collection_error(collection, detail = str(exc))
                raise HTTPException(status_code = 500, detail = str(exc)) from exc

    def reindex_collection(self, collection_id: str, initiator: str = "admin") -> RAGJobSummary:
        with self._lock:
            collection = self._require_collection(collection_id)
            try:
                connection = self._require_connection_profile(collection["connection_profile_id"])
                ingestion = self._require_ingestion_profile(collection["ingestion_profile_id"])
                dataset_ids = self._repository.list_collection_dataset_ids(collection_id)
                if not dataset_ids:
                    raise HTTPException(
                        status_code = 400,
                        detail = "Cannot reindex a collection without linked datasets.",
                    )

                documents: list[dict] = []
                for dataset_id in dataset_ids:
                    self._require_dataset(dataset_id)
                    documents.extend(self._repository.list_dataset_documents(dataset_id))

                chunk_texts = self._materialize_chunk_texts(documents, ingestion)
                components = self._ingestion_components(ingestion)
                vectors = self._embedding_provider.embed_texts(components["embedding_model"], chunk_texts)
                if not vectors:
                    raise HTTPException(status_code = 400, detail = "No chunks generated for reindex.")

                adapter = self._vector_store_factory(connection)
                current_projection = self._require_projection(collection["active_projection_id"])
                next_projection = self._build_projection(
                    collection_id = collection_id,
                    remote_collection_name = str(collection["remote_collection_name"]),
                    backend = collection["backend"],
                    extractor = components["extractor"],
                    ocr_engine = components["ocr_engine"],
                    embedder = str(ingestion["embedder"]),
                    embedding_model = components["embedding_model"],
                    chunk_size = int(ingestion["chunk_size"]),
                    chunk_overlap = int(ingestion["chunk_overlap"]),
                    version = int(current_projection["version"]) + 1,
                    status = "ready",
                    indexed_at = _now_iso(),
                    source_document_count = len(documents),
                )
                points, total_chunks = build_qdrant_points(
                    collection_id = collection_id,
                    projection_id = str(next_projection["id"]),
                    documents = documents,
                    chunk_size = int(ingestion["chunk_size"]),
                    chunk_overlap = int(ingestion["chunk_overlap"]),
                    extractor = components["extractor"],
                    ocr_engine = components["ocr_engine"],
                    chunk_recipe = str(next_projection["chunk_recipe"]),
                    embedding_config = self._embedding_config(components["embedding_model"]),
                    vectors = vectors,
                )
                adapter.ensure_collection(
                    collection_name = str(next_projection["physical_collection_name"]),
                    vector_size = len(vectors[0]),
                )
                adapter.upsert_chunks(
                    collection_name = str(next_projection["physical_collection_name"]),
                    points = points,
                )
                self._repository.upsert_projection(next_projection)
                collection["active_projection_id"] = next_projection["id"]
                collection["last_reindex_at"] = _now_iso()
                collection["last_job_status"] = "completed"
                collection["sync_status"] = "pending"
                collection["last_error"] = None
                collection["chunks_count"] = total_chunks
                self._repository.upsert_collection(collection)
                job = self._register_job(
                    job_type = "reindex",
                    collection = collection,
                    initiator = initiator,
                    dataset = None,
                    stage_label = "Rebuilt active projection",
                )
                return RAGJobSummary.model_validate(job)
            except HTTPException as exc:
                self._mark_collection_error(collection, detail = exc.detail)
                raise
            except Exception as exc:
                self._mark_collection_error(collection, detail = str(exc))
                raise HTTPException(status_code = 500, detail = str(exc)) from exc

    def sync_open_webui(self, collection_id: str, initiator: str = "admin") -> RAGJobSummary:
        with self._lock:
            collection = self._require_collection(collection_id)
            collection["sync_status"] = "synced"
            collection["last_job_status"] = "completed"
            collection["last_error"] = None
            self._repository.upsert_collection(collection)
            job = self._register_job(
                job_type = "sync-open-webui",
                collection = collection,
                initiator = initiator,
                dataset = None,
                stage_label = "Marked collection compatible with Open WebUI knowledge",
            )
            return RAGJobSummary.model_validate(job)

    def list_jobs(self) -> RAGJobListResponse:
        items = [RAGJobSummary.model_validate(item) for item in self._repository.list_jobs()]
        items.sort(key = lambda item: item.created_at, reverse = True)
        return RAGJobListResponse(items = items)

    def get_diagnostics(self) -> RAGDiagnosticsResponse:
        diagnostics: list[RAGConnectionDiagnostic] = []
        for profile in self._repository.list_connection_profiles():
            if not profile.get("enabled", True):
                diagnostics.append(
                    RAGConnectionDiagnostic(
                        id = profile["id"],
                        name = profile["name"],
                        backend = profile["backend"],
                        status = "disabled",
                        details = {},
                    )
                )
                continue
            status_payload = self._vector_store_factory(profile).health_check()
            diagnostics.append(
                RAGConnectionDiagnostic(
                    id = profile["id"],
                    name = profile["name"],
                    backend = profile["backend"],
                    status = str(status_payload.get("status", "unknown")),
                    details = dict(status_payload.get("details", {})),
                )
            )
        return RAGDiagnosticsResponse(
            connection_profiles = diagnostics,
            collections = self.list_collections().items,
        )

    def _to_connection_summary(self, record: dict) -> RAGConnectionProfileSummary:
        payload = dict(record)
        payload.pop("api_key", None)
        return RAGConnectionProfileSummary.model_validate(payload)

    def _to_collection_summary(self, record: dict) -> RAGCollectionSummary:
        connection = self._require_connection_profile(record["connection_profile_id"])
        ingestion = self._require_ingestion_profile(record["ingestion_profile_id"])
        payload = dict(record)
        payload["connection_profile_name"] = connection["name"]
        payload["ingestion_profile_name"] = ingestion["name"]
        payload["active_projection"] = self._require_projection(record["active_projection_id"])
        return RAGCollectionSummary.model_validate(payload)

    def _materialize_chunk_texts(self, documents: list[dict], ingestion_profile: dict) -> list[str]:
        chunk_texts: list[str] = []
        for document in documents:
            extracted_path = Path(str(document["extracted_path"]))
            if not extracted_path.exists():
                raise HTTPException(
                    status_code = 422,
                    detail = f"Missing extracted text artifact for document `{document['document_name']}`.",
                )
            chunk_texts.extend(
                chunk_text(
                    read_extracted_text(extracted_path),
                    chunk_size = int(ingestion_profile["chunk_size"]),
                    chunk_overlap = int(ingestion_profile["chunk_overlap"]),
                )
            )
        return [chunk.text for chunk in chunk_texts]

    def _refresh_dataset_stats(self, dataset_id: str, chunk_count: int | None = None) -> None:
        dataset = self._require_dataset(dataset_id)
        documents = self._repository.list_dataset_documents(dataset_id)
        total_characters = sum(int(item.get("text_char_count", 0)) for item in documents)
        dataset["documents_count"] = len(documents)
        dataset["total_characters"] = total_characters
        if chunk_count is not None:
            dataset["chunks_count"] = chunk_count
        dataset["status"] = "ready" if documents else "empty"
        dataset["updated_at"] = _now_iso()
        dataset["last_error"] = None
        self._repository.upsert_dataset(dataset)

    def _refresh_collection_stats(self, collection_id: str) -> None:
        collection = self._require_collection(collection_id)
        dataset_ids = self._repository.list_collection_dataset_ids(collection_id)
        datasets = [self._require_dataset(dataset_id) for dataset_id in dataset_ids]
        collection["documents_count"] = sum(int(item.get("documents_count", 0)) for item in datasets)
        collection["chunks_count"] = sum(int(item.get("chunks_count", 0)) for item in datasets)
        self._repository.upsert_collection(collection)

    def _build_projection(
        self,
        *,
        collection_id: str,
        remote_collection_name: str,
        backend: str,
        extractor: str,
        ocr_engine: str,
        embedder: str,
        embedding_model: str | None,
        chunk_size: int,
        chunk_overlap: int,
        version: int,
        status: str,
        indexed_at: str | None,
        source_document_count: int,
    ) -> dict:
        return {
            "id": f"{collection_id}:{backend}:v{version}",
            "collection_id": collection_id,
            "backend": backend,
            "embedder": embedder,
            "embedding_model": embedding_model or embedder,
            "physical_collection_name": f"{remote_collection_name}__proj_v{version}",
            "extractor": extractor,
            "ocr_engine": ocr_engine,
            "chunk_recipe": f"char:{chunk_size}:{chunk_overlap}",
            "status": status,
            "version": version,
            "source_document_count": source_document_count,
            "indexed_at": indexed_at,
        }

    def _require_projection(self, projection_id: str) -> dict:
        record = self._repository.get_projection(projection_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG index projection not found.")
        return record

    def _mark_collection_error(self, collection: dict, *, detail: object) -> None:
        collection["last_job_status"] = "error"
        collection["last_error"] = str(detail)
        collection["sync_status"] = "error"
        self._repository.upsert_collection(collection)

    def _embedding_config(self, model_id: str) -> dict[str, object]:
        return {
            "engine": str(getattr(self._embedding_provider, "engine", "unknown")),
            "model": model_id,
        }

    def _module_action_response(
        self,
        summary: RAGModuleInstallationSummary,
        *,
        message: str,
    ) -> RAGModuleActionResponse:
        return RAGModuleActionResponse(
            module_id = summary.module_id,
            status = summary.status,
            message = message,
            details = summary.model_dump(),
        )

    def _ingestion_components(self, ingestion: dict) -> dict[str, str]:
        extractor = self._resolve_ingestion_component(
            ingestion,
            component_key = "extractor",
            instance_key = "extractor_instance_id",
            enabled_key = "extractor_enabled",
            disabled_value = "none",
        )
        ocr_engine = self._resolve_ingestion_component(
            ingestion,
            component_key = "ocr_engine",
            instance_key = "ocr_instance_id",
            enabled_key = "ocr_enabled",
            disabled_value = "none",
        )
        embedding_model = self._resolve_ingestion_component(
            ingestion,
            component_key = "embedder",
            instance_key = "embedder_instance_id",
            enabled_key = None,
            disabled_value = "",
        )
        reranker = self._resolve_ingestion_component(
            ingestion,
            component_key = "reranker",
            instance_key = "reranker_instance_id",
            enabled_key = "reranker_enabled",
            disabled_value = "none",
        )
        return {
            "extractor": extractor,
            "ocr_engine": ocr_engine,
            "embedding_model": embedding_model,
            "reranker": reranker,
        }

    def _resolve_ingestion_component(
        self,
        ingestion: dict,
        *,
        component_key: str,
        instance_key: str,
        enabled_key: str | None,
        disabled_value: str,
    ) -> str:
        if enabled_key is not None and not bool(ingestion.get(enabled_key, True)):
            return disabled_value
        fallback = str(ingestion.get(component_key) or disabled_value)
        instance_id = ingestion.get(instance_key)
        if not instance_id:
            return fallback
        instance = self._require_module_instance(str(instance_id))
        return str(
            instance.get("local_path")
            or instance.get("model_id")
            or instance.get("service_url")
            or instance.get("binary_path")
            or instance.get("module_id")
            or fallback
        )

    def _validate_module_instance_kind(self, instance_id: str | None, expected_kind: str) -> None:
        if not instance_id:
            return
        instance = self._require_module_instance(instance_id)
        if instance.get("kind") != expected_kind:
            raise HTTPException(
                status_code = 400,
                detail = f"Module instance `{instance_id}` is not a `{expected_kind}` module.",
            )

    def _require_module(self, module_id: str) -> RAGModuleCatalogItem:
        record = get_rag_module_by_id(module_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG module not found.")
        return record

    def _require_module_instance(self, instance_id: str) -> dict:
        record = self._repository.get_module_instance(instance_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG module instance not found.")
        return record

    def _register_job(
        self,
        *,
        job_type: str,
        collection: dict,
        initiator: str,
        dataset: dict | None,
        stage_label: str,
    ) -> dict:
        now = _now_iso()
        job = {
            "id": _new_id("ragjob"),
            "job_type": job_type,
            "status": "completed",
            "collection_id": collection["id"],
            "collection_name": collection["name"],
            "dataset_id": dataset["id"] if dataset else None,
            "dataset_name": dataset["name"] if dataset else None,
            "initiator": initiator,
            "stage_label": stage_label,
            "progress_percent": 100,
            "duration_seconds": 1,
            "warning_count": 0,
            "created_at": now,
            "finished_at": now,
        }
        self._repository.insert_job(job)
        return job

    def _dataset_dir(self, dataset_id: str) -> Path:
        return ensure_dir(self._asset_root / dataset_id)

    def _require_connection_profile(self, profile_id: str) -> dict:
        record = self._repository.get_connection_profile(profile_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG connection profile not found.")
        return record

    def _require_ingestion_profile(self, profile_id: str) -> dict:
        record = self._repository.get_ingestion_profile(profile_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG ingestion profile not found.")
        return record

    def _require_dataset(self, dataset_id: str) -> dict:
        record = self._repository.get_dataset(dataset_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG dataset not found.")
        return record

    def _require_collection(self, collection_id: str) -> dict:
        record = self._repository.get_collection(collection_id)
        if record is None:
            raise HTTPException(status_code = 404, detail = "RAG collection not found.")
        return record


_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def reset_rag_service(service: RAGService | None = None) -> None:
    global _rag_service
    _rag_service = service if service is not None else RAGService()
