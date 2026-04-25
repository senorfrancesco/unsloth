# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from core.rag.documents import (
    build_qdrant_points,
    chunk_text,
    read_extracted_text,
    store_uploaded_documents,
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
    uninstall_catalog_module,
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
    RAGCollectionChunkPayload,
    RAGCollectionChunkStats,
    RAGCollectionDistributionItem,
    RAGCollectionInspectDistributions,
    RAGCollectionInspectResponse,
    RAGCollectionInspectVectorState,
    RAGCollectionSampleChunksResponse,
    RAGCollectionSearchResponse,
    RAGCollectionSearchResult,
    RAGCollectionSummary,
    RAGConnectionDiagnostic,
    RAGConnectionProfileListResponse,
    RAGConnectionProfileSummary,
    RAGDatasetDetailResponse,
    RAGDatasetDocumentSummary,
    RAGDatasetListResponse,
    RAGDatasetSummary,
    RAGDiagnosticsResponse,
    RAGDocumentUploadItem,
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
    SearchRAGCollectionRequest,
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

    def uninstall_module_package(self, module_id: str) -> RAGModuleActionResponse:
        catalog_item = self._require_module(module_id)
        if not catalog_item.package_name:
            raise HTTPException(
                status_code = 400,
                detail = "This module does not declare a Python package to uninstall.",
            )
        summary = uninstall_catalog_module(catalog_item)
        affected_modules: list[dict[str, Any]] = []
        for item in get_rag_module_catalog().items:
            if item.package_name != catalog_item.package_name:
                continue
            affected_summary = check_catalog_module(item)
            self._repository.upsert_module_installation(affected_summary.model_dump())
            affected_modules.append(affected_summary.model_dump())
        response = self._module_action_response(
            summary,
            message = "Managed module uninstall completed.",
        )
        response.details["affected_modules"] = affected_modules
        return response

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

    def delete_module_instance(self, instance_id: str) -> RAGModuleActionResponse:
        record = self._require_module_instance(instance_id)
        cleared_profiles = self._clear_module_instance_references(instance_id)
        deleted = self._repository.delete_module_instance(instance_id)
        if not deleted:
            raise HTTPException(status_code = 404, detail = "RAG module instance not found.")
        return RAGModuleActionResponse(
            module_id = str(record["module_id"]),
            status = "configured",
            message = "Module instance record deleted.",
            details = {
                "instance_id": instance_id,
                "deleted_files": False,
                "cleared_ingestion_profile_ids": cleared_profiles,
            },
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
                "processing_status": "ready",
                "processing_error": None,
                "processed_at": _now_iso(),
                "extractor": "manual-text",
                "ocr_engine": "none",
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
            self._mark_dataset_processing(dataset_id)
            try:
                stored_documents = await store_uploaded_documents(dataset_dir = dataset_dir, file = file)
            except HTTPException as exc:
                self._mark_dataset_error(dataset_id, detail = exc.detail)
                raise
            except Exception as exc:
                self._mark_dataset_error(dataset_id, detail = str(exc))
                raise HTTPException(status_code = 422, detail = str(exc)) from exc
            processed_at = _now_iso()
            documents: list[dict[str, object]] = []
            for stored in stored_documents:
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
                    "created_at": processed_at,
                    "processing_status": "ready",
                    "processing_error": None,
                    "processed_at": processed_at,
                    "extractor": "builtin-file",
                    "ocr_engine": "none",
                }
                self._repository.insert_dataset_document(document)
                documents.append(document)
            self._refresh_dataset_stats(dataset_id)
            first_document = documents[0]
            uploaded_items = [
                RAGDocumentUploadItem(
                    file_id = str(document["id"]),
                    filename = str(document["document_name"]),
                    size_bytes = int(document["size_bytes"]),
                    status = "ok",
                )
                for document in documents
            ]
            return RAGDocumentUploadResponse(
                file_id = str(first_document["id"]),
                filename = str(first_document["document_name"]),
                size_bytes = sum(int(document["size_bytes"]) for document in documents),
                status = "ok",
                files = uploaded_items,
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
                indexed_at = _now_iso()
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
                    indexed_at = indexed_at,
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
                projection["indexed_at"] = indexed_at
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
                indexed_at = _now_iso()
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
                    indexed_at = indexed_at,
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
                    indexed_at = indexed_at,
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

    def inspect_collection(self, collection_id: str) -> RAGCollectionInspectResponse:
        collection, connection, ingestion, projection, adapter = self._qdrant_inspection_context(collection_id)
        warnings: list[str] = []
        physical_collection_name = str(projection["physical_collection_name"])

        health = self._safe_vector_store_health(adapter)
        qdrant_status = str(health.get("status", "unknown"))
        qdrant_details = dict(health.get("details", {}))
        collection_info: dict[str, Any] = {}
        points: list[dict[str, Any]] = []
        truncated = False

        if qdrant_status == "healthy":
            info = self._safe_collection_info(adapter, physical_collection_name)
            info_status = str(info.get("status", "unknown"))
            collection_info = dict(info.get("details", {}))
            if info_status == "missing":
                qdrant_status = "missing"
                warnings.append(
                    f"Physical Qdrant collection `{physical_collection_name}` was not found."
                )
            elif info_status == "error":
                qdrant_status = "error"
                warnings.append(str(info.get("message", "Failed to read Qdrant collection info.")))
            else:
                points, truncated = self._scroll_collection_points_for_inspect(
                    adapter,
                    collection_name = physical_collection_name,
                    limit = 5000,
                    warnings = warnings,
                )
        else:
            warnings.append(
                "Qdrant connection is not healthy; collection payload could not be inspected."
            )

        if collection.get("last_error"):
            warnings.append(str(collection["last_error"]))

        chunks = [self._chunk_payload_from_point(point, projection = projection) for point in points]
        qdrant_points_total = self._extract_qdrant_points_total(collection_info)
        stats = self._build_chunk_stats(
            chunks,
            points = points,
            projection = projection,
            qdrant_points_total = qdrant_points_total,
            inspected_limit = 5000,
            truncated = truncated,
        )
        if (
            not truncated
            and qdrant_status == "healthy"
            and int(collection.get("chunks_count", 0)) != stats.chunks_total
        ):
            warnings.append(
                "Stored collection chunk count does not match the active Qdrant projection."
            )

        return RAGCollectionInspectResponse(
            collection = self._to_collection_summary(collection),
            connection_profile = self._to_connection_summary(connection),
            ingestion_profile = RAGIngestionProfileSummary.model_validate(ingestion),
            active_projection = RAGIndexProjectionSummary.model_validate(projection),
            qdrant = RAGCollectionInspectVectorState(
                backend = "qdrant",
                status = qdrant_status,
                physical_collection_name = physical_collection_name,
                details = qdrant_details,
                collection_info = collection_info,
            ),
            stats = stats,
            distributions = self._build_chunk_distributions(chunks),
            warnings = warnings,
        )

    def sample_collection_chunks(
        self,
        collection_id: str,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> RAGCollectionSampleChunksResponse:
        _collection, _connection, _ingestion, projection, adapter = self._qdrant_inspection_context(collection_id)
        physical_collection_name = str(projection["physical_collection_name"])
        response = self._call_vector_store(
            lambda: adapter.scroll_points(
                collection_name = physical_collection_name,
                limit = limit,
                offset = offset,
            ),
            action_label = "read Qdrant collection chunks",
        )
        points = [dict(point) for point in response.get("points", []) if isinstance(point, dict)]
        next_page_offset = response.get("next_page_offset")
        next_offset = int(next_page_offset) if isinstance(next_page_offset, int) else None
        return RAGCollectionSampleChunksResponse(
            collection_id = collection_id,
            limit = limit,
            offset = offset,
            next_offset = next_offset,
            items = [
                self._chunk_payload_from_point(point, projection = projection)
                for point in points
            ],
        )

    def search_collection(
        self,
        collection_id: str,
        payload: SearchRAGCollectionRequest,
    ) -> RAGCollectionSearchResponse:
        _collection, _connection, _ingestion, projection, adapter = self._qdrant_inspection_context(collection_id)
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code = 400, detail = "Search query cannot be empty.")

        embedding_model = str(projection["embedding_model"])
        try:
            query_vectors = self._embedding_provider.embed_texts(embedding_model, [query])
        except Exception as exc:
            raise HTTPException(
                status_code = 500,
                detail = f"Failed to build query embedding: {exc}",
            ) from exc
        if not query_vectors:
            raise HTTPException(status_code = 400, detail = "Search query produced no embedding.")

        physical_collection_name = str(projection["physical_collection_name"])
        response = self._call_vector_store(
            lambda: adapter.search_points(
                collection_name = physical_collection_name,
                vector = query_vectors[0],
                limit = payload.limit,
            ),
            action_label = "search Qdrant collection",
        )
        raw_points = [dict(point) for point in response.get("points", []) if isinstance(point, dict)]
        results: list[RAGCollectionSearchResult] = []
        for point in raw_points:
            chunk = self._chunk_payload_from_point(point, projection = projection)
            results.append(
                RAGCollectionSearchResult(
                    **chunk.model_dump(),
                    score = float(point.get("score", 0.0)),
                )
            )

        return RAGCollectionSearchResponse(
            collection_id = collection_id,
            query = query,
            limit = payload.limit,
            embedding_model = embedding_model,
            results = results,
        )

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

    def _qdrant_inspection_context(
        self,
        collection_id: str,
    ) -> tuple[dict, dict, dict, dict, VectorStoreAdapterProtocol]:
        collection = self._require_collection(collection_id)
        connection = self._require_connection_profile(collection["connection_profile_id"])
        ingestion = self._require_ingestion_profile(collection["ingestion_profile_id"])
        projection = self._require_projection(collection["active_projection_id"])
        backend_ids = {
            str(collection.get("backend")),
            str(connection.get("backend")),
            str(projection.get("backend")),
        }
        if backend_ids != {"qdrant"}:
            raise HTTPException(
                status_code = 400,
                detail = "RAG collection inspector currently supports Qdrant collections only.",
            )
        try:
            adapter = self._vector_store_factory(connection)
        except Exception as exc:
            raise HTTPException(
                status_code = 400,
                detail = f"Failed to initialize Qdrant adapter: {exc}",
            ) from exc
        return collection, connection, ingestion, projection, adapter

    def _safe_vector_store_health(self, adapter: VectorStoreAdapterProtocol) -> dict[str, Any]:
        try:
            return dict(adapter.health_check())
        except Exception as exc:
            return {"status": "error", "details": {"message": str(exc)}}

    def _safe_collection_info(
        self,
        adapter: VectorStoreAdapterProtocol,
        collection_name: str,
    ) -> dict[str, Any]:
        try:
            return dict(adapter.get_collection_info(collection_name = collection_name))
        except Exception as exc:
            return {"status": "error", "message": str(exc), "details": {}}

    def _scroll_collection_points_for_inspect(
        self,
        adapter: VectorStoreAdapterProtocol,
        *,
        collection_name: str,
        limit: int,
        warnings: list[str],
    ) -> tuple[list[dict[str, Any]], bool]:
        points: list[dict[str, Any]] = []
        cursor: object | None = None
        while len(points) < limit:
            page_size = min(256, limit - len(points))
            try:
                response = adapter.scroll_points(
                    collection_name = collection_name,
                    limit = page_size,
                    offset = cursor,
                )
            except Exception as exc:
                warnings.append(f"Failed to scroll Qdrant collection payload: {exc}")
                break
            batch = [
                dict(point)
                for point in response.get("points", [])
                if isinstance(point, dict)
            ]
            points.extend(batch)
            cursor = response.get("next_page_offset")
            if not batch or cursor is None:
                break
        return points, cursor is not None

    def _call_vector_store(
        self,
        action: Callable[[], dict[str, object]],
        *,
        action_label: str,
    ) -> dict[str, object]:
        try:
            return action()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code = 503,
                detail = f"Failed to {action_label}: {exc}",
            ) from exc

    def _chunk_payload_from_point(
        self,
        point: dict[str, Any],
        *,
        projection: dict,
    ) -> RAGCollectionChunkPayload:
        payload = point.get("payload") if isinstance(point.get("payload"), dict) else {}
        payload = dict(payload)
        indexed_at = payload.get("indexed_at") or projection.get("indexed_at")
        payload_for_ui = dict(payload)
        if indexed_at and not payload_for_ui.get("indexed_at"):
            payload_for_ui["indexed_at"] = indexed_at
        embedding_config = payload.get("embedding_config")
        return RAGCollectionChunkPayload(
            point_id = str(point.get("id") or ""),
            text = str(payload.get("text") or ""),
            file_id = self._optional_str(payload.get("file_id") or payload.get("knowledge_file_id")),
            document_id = self._optional_str(payload.get("document_id")),
            source = self._optional_str(
                payload.get("source") or payload.get("document_name") or payload.get("name")
            ),
            hash = self._optional_str(payload.get("hash") or payload.get("source_hash")),
            extractor = self._optional_str(payload.get("extractor") or projection.get("extractor")),
            ocr_engine = self._optional_str(payload.get("ocr_engine") or projection.get("ocr_engine")),
            embedding_config = embedding_config if isinstance(embedding_config, dict) else {},
            chunk_recipe = self._optional_str(payload.get("chunk_recipe") or projection.get("chunk_recipe")),
            indexed_at = self._optional_str(indexed_at),
            chunk_index = self._optional_int(payload.get("chunk_index")),
            payload = payload_for_ui,
        )

    def _build_chunk_stats(
        self,
        chunks: list[RAGCollectionChunkPayload],
        *,
        points: list[dict[str, Any]],
        projection: dict,
        qdrant_points_total: int | None,
        inspected_limit: int,
        truncated: bool,
    ) -> RAGCollectionChunkStats:
        text_lengths = [len(chunk.text) for chunk in chunks if chunk.text]
        document_ids = {
            chunk.document_id or chunk.file_id
            for chunk in chunks
            if chunk.document_id or chunk.file_id
        }
        missing_indexed_at_count = 0
        for point in points:
            payload = point.get("payload") if isinstance(point.get("payload"), dict) else {}
            if not payload.get("indexed_at") and not projection.get("indexed_at"):
                missing_indexed_at_count += 1
        return RAGCollectionChunkStats(
            chunks_total = len(chunks),
            qdrant_points_total = qdrant_points_total,
            documents_total = len(document_ids),
            average_chunk_chars = round(sum(text_lengths) / len(text_lengths), 1) if text_lengths else 0.0,
            min_chunk_chars = min(text_lengths) if text_lengths else 0,
            max_chunk_chars = max(text_lengths) if text_lengths else 0,
            missing_text_count = sum(1 for chunk in chunks if not chunk.text),
            missing_indexed_at_count = missing_indexed_at_count,
            inspected_limit = inspected_limit,
            truncated = truncated,
        )

    def _build_chunk_distributions(
        self,
        chunks: list[RAGCollectionChunkPayload],
    ) -> RAGCollectionInspectDistributions:
        document_counts: dict[str, dict[str, object]] = {}
        for chunk in chunks:
            document_id = chunk.document_id or chunk.file_id or "unknown"
            label = chunk.source or document_id
            current = document_counts.setdefault(
                document_id,
                {"label": label, "count": 0},
            )
            current["count"] = int(current["count"]) + 1

        bucket_counts = {
            "missing": 0,
            "0-199": 0,
            "200-399": 0,
            "400-799": 0,
            "800-1199": 0,
            "1200+": 0,
        }
        for chunk in chunks:
            length = len(chunk.text)
            if length == 0:
                bucket_counts["missing"] += 1
            elif length < 200:
                bucket_counts["0-199"] += 1
            elif length < 400:
                bucket_counts["200-399"] += 1
            elif length < 800:
                bucket_counts["400-799"] += 1
            elif length < 1200:
                bucket_counts["800-1199"] += 1
            else:
                bucket_counts["1200+"] += 1

        status_counts = {"indexed": 0, "missing indexed_at": 0}
        for chunk in chunks:
            if chunk.indexed_at:
                status_counts["indexed"] += 1
            else:
                status_counts["missing indexed_at"] += 1

        documents = [
            RAGCollectionDistributionItem(
                id = document_id,
                label = str(payload["label"]),
                count = int(payload["count"]),
                value = int(payload["count"]),
            )
            for document_id, payload in sorted(
                document_counts.items(),
                key = lambda item: (-int(item[1]["count"]), str(item[1]["label"])),
            )[:20]
        ]
        chunk_sizes = [
            RAGCollectionDistributionItem(
                id = bucket,
                label = bucket,
                count = count,
                value = count,
            )
            for bucket, count in bucket_counts.items()
            if count > 0
        ]
        indexing_statuses = [
            RAGCollectionDistributionItem(
                id = status,
                label = status,
                count = count,
                value = count,
            )
            for status, count in status_counts.items()
            if count > 0
        ]
        return RAGCollectionInspectDistributions(
            documents = documents,
            chunk_sizes = chunk_sizes,
            indexing_statuses = indexing_statuses,
        )

    def _extract_qdrant_points_total(self, collection_info: dict[str, Any]) -> int | None:
        for key in ("points_count", "vectors_count", "indexed_vectors_count"):
            value = collection_info.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        return None

    @staticmethod
    def _optional_str(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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

    def _mark_dataset_processing(self, dataset_id: str) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset["status"] = "processing"
        dataset["last_error"] = None
        dataset["updated_at"] = _now_iso()
        self._repository.upsert_dataset(dataset)

    def _mark_dataset_error(self, dataset_id: str, *, detail: object) -> None:
        dataset = self._require_dataset(dataset_id)
        dataset["status"] = "error"
        dataset["last_error"] = str(detail)
        dataset["updated_at"] = _now_iso()
        self._repository.upsert_dataset(dataset)

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

    def _clear_module_instance_references(self, instance_id: str) -> list[str]:
        instance_fields = (
            "extractor_instance_id",
            "ocr_instance_id",
            "embedder_instance_id",
            "reranker_instance_id",
        )
        cleared_profile_ids: list[str] = []
        for profile in self._repository.list_ingestion_profiles():
            changed = False
            for field in instance_fields:
                if profile.get(field) == instance_id:
                    profile[field] = None
                    changed = True
            if changed:
                self._repository.upsert_ingestion_profile(profile)
                cleared_profile_ids.append(str(profile["id"]))
        return cleared_profile_ids

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
