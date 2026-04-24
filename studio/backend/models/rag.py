# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


RAGBackendId = Literal["qdrant", "pgvector"]
RAGProviderStage = Literal["first-class", "design-only", "future"]
RAGCollectionSyncStatus = Literal["not-synced", "pending", "synced", "error"]
RAGJobType = Literal["publish", "reindex", "sync-open-webui"]
RAGJobStatus = Literal["queued", "running", "completed", "error"]
RAGSourceKind = Literal["documents", "normalized-text"]
RAGDatasetStatus = Literal["empty", "ready", "processing", "error"]
RAGConnectionStatus = Literal["healthy", "error", "disabled", "unknown"]
RAGDocumentUploadStatus = Literal["ok", "error"]
RAGModuleKind = Literal["extractor", "ocr", "embedder", "reranker", "chunker", "vector_store"]
RAGModuleSourceType = Literal[
    "builtin",
    "python_package",
    "system_binary",
    "local_model_path",
    "service_url",
    "wheel",
    "wheelhouse",
]
RAGModuleStatus = Literal["available", "missing", "configured", "error", "disabled"]


class RAGProviderOption(BaseModel):
    id: str
    label: str
    description: str
    enabled: bool = True
    stage: RAGProviderStage
    kind: RAGModuleKind | None = None
    source_type: RAGModuleSourceType | None = None
    package_name: str | None = None
    module_name: str | None = None
    binary_name: str | None = None
    default_model_id: str | None = None
    installable: bool = False
    configurable: bool = False


class RAGProvidersResponse(BaseModel):
    vector_stores: list[RAGProviderOption]
    extractors: list[RAGProviderOption]
    ocr_engines: list[RAGProviderOption]
    embedders: list[RAGProviderOption]
    rerankers: list[RAGProviderOption]


class RAGModuleCatalogItem(BaseModel):
    id: str
    kind: RAGModuleKind
    label: str
    description: str
    enabled: bool = True
    stage: RAGProviderStage
    source_type: RAGModuleSourceType
    package_name: str | None = None
    module_name: str | None = None
    binary_name: str | None = None
    default_model_id: str | None = None
    installable: bool = False
    configurable: bool = False
    dependencies: list[str] = Field(default_factory = list)
    config_schema: dict[str, Any] = Field(default_factory = dict)


class RAGModuleCatalogResponse(BaseModel):
    items: list[RAGModuleCatalogItem]


class RAGModuleInstallationSummary(BaseModel):
    module_id: str
    kind: RAGModuleKind
    status: RAGModuleStatus
    source_type: RAGModuleSourceType
    version: str | None = None
    path: str | None = None
    last_checked_at: str | None = None
    last_error: str | None = None
    install_command: str | None = None


class RAGModuleInstallationListResponse(BaseModel):
    items: list[RAGModuleInstallationSummary]


class RAGModuleInstanceSummary(BaseModel):
    id: str
    name: str
    module_id: str
    kind: RAGModuleKind
    source_type: RAGModuleSourceType
    model_id: str | None = None
    local_path: str | None = None
    service_url: str | None = None
    binary_path: str | None = None
    enabled: bool = True
    status: RAGModuleStatus = "configured"
    last_checked_at: str | None = None
    last_error: str | None = None
    created_at: str


class RAGModuleInstanceListResponse(BaseModel):
    items: list[RAGModuleInstanceSummary]


class InstallRAGModuleRequest(BaseModel):
    package_path: str | None = Field(default = None, max_length = 1000)
    wheelhouse_path: str | None = Field(default = None, max_length = 1000)


class CreateRAGModuleInstanceRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 120)
    module_id: str = Field(min_length = 1, max_length = 120)
    source_type: RAGModuleSourceType
    model_id: str | None = Field(default = None, max_length = 500)
    local_path: str | None = Field(default = None, max_length = 1000)
    service_url: str | None = Field(default = None, max_length = 1000)
    binary_path: str | None = Field(default = None, max_length = 1000)
    enabled: bool = True


class RAGModuleActionResponse(BaseModel):
    module_id: str
    status: RAGModuleStatus
    message: str
    details: dict[str, Any] = Field(default_factory = dict)


class RAGIndexProjectionSummary(BaseModel):
    id: str
    backend: RAGBackendId
    embedder: str
    embedding_model: str
    physical_collection_name: str
    extractor: str
    ocr_engine: str
    chunk_recipe: str
    status: str
    version: int
    source_document_count: int = 0
    indexed_at: str | None = None


class RAGConnectionProfileSummary(BaseModel):
    id: str
    name: str
    backend: RAGBackendId
    base_url: str
    default_collection_prefix: str | None = None
    enabled: bool = True
    status: RAGConnectionStatus = "unknown"
    created_at: str


class RAGConnectionProfileListResponse(BaseModel):
    items: list[RAGConnectionProfileSummary]


class RAGIngestionProfileSummary(BaseModel):
    id: str
    name: str
    extractor: str
    ocr_engine: str
    embedder: str
    reranker: str | None = None
    extractor_enabled: bool = True
    ocr_enabled: bool = True
    reranker_enabled: bool = False
    extractor_instance_id: str | None = None
    ocr_instance_id: str | None = None
    embedder_instance_id: str | None = None
    reranker_instance_id: str | None = None
    chunk_size: int
    chunk_overlap: int
    created_at: str


class RAGIngestionProfileListResponse(BaseModel):
    items: list[RAGIngestionProfileSummary]


class RAGDatasetDocumentSummary(BaseModel):
    id: str
    dataset_id: str
    document_name: str
    mime_type: str
    source_kind: RAGSourceKind
    size_bytes: int
    text_char_count: int
    content_hash: str
    created_at: str


class RAGDatasetSummary(BaseModel):
    id: str
    name: str
    source_kind: RAGSourceKind
    description: str | None = None
    status: RAGDatasetStatus
    documents_count: int
    chunks_count: int
    total_characters: int
    last_error: str | None = None
    created_at: str
    updated_at: str


class RAGDatasetListResponse(BaseModel):
    items: list[RAGDatasetSummary]


class RAGDatasetDetailResponse(BaseModel):
    dataset: RAGDatasetSummary
    documents: list[RAGDatasetDocumentSummary]


class RAGCollectionSummary(BaseModel):
    id: str
    name: str
    backend: RAGBackendId
    connection_profile_id: str
    connection_profile_name: str
    ingestion_profile_id: str
    ingestion_profile_name: str
    remote_collection_name: str
    documents_count: int
    chunks_count: int
    sync_status: RAGCollectionSyncStatus
    last_job_status: RAGJobStatus | None = None
    last_error: str | None = None
    last_reindex_at: str | None = None
    active_projection: RAGIndexProjectionSummary


class RAGCollectionListResponse(BaseModel):
    items: list[RAGCollectionSummary]


class RAGJobSummary(BaseModel):
    id: str
    job_type: RAGJobType
    status: RAGJobStatus
    collection_id: str
    collection_name: str
    dataset_id: str | None = None
    dataset_name: str | None = None
    initiator: str
    stage_label: str
    progress_percent: int = Field(ge = 0, le = 100)
    duration_seconds: int | None = None
    warning_count: int = 0
    created_at: str
    finished_at: str | None = None


class RAGJobListResponse(BaseModel):
    items: list[RAGJobSummary]


class RAGOverviewBackend(BaseModel):
    id: RAGBackendId
    enabled: bool
    stage: RAGProviderStage


class RAGOverviewResponse(BaseModel):
    collections_total: int
    jobs_total: int
    datasets_total: int
    documents_total: int
    chunks_total: int
    synced_collections_total: int
    supported_backends: list[RAGOverviewBackend]


class RAGConnectionDiagnostic(BaseModel):
    id: str
    name: str
    backend: RAGBackendId
    status: RAGConnectionStatus
    details: dict[str, Any] = Field(default_factory = dict)


class RAGDiagnosticsResponse(BaseModel):
    connection_profiles: list[RAGConnectionDiagnostic]
    collections: list[RAGCollectionSummary]


class CreateRAGConnectionProfileRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 120)
    backend: RAGBackendId
    base_url: str = Field(min_length = 1, max_length = 500)
    api_key: str | None = Field(default = None, max_length = 500)
    default_collection_prefix: str | None = Field(default = None, max_length = 120)
    enabled: bool = True


class CreateRAGIngestionProfileRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 120)
    extractor: str = Field(default = "builtin-text", min_length = 1, max_length = 60)
    ocr_engine: str = Field(default = "none", min_length = 1, max_length = 60)
    embedder: str = Field(min_length = 1, max_length = 120)
    reranker: str | None = Field(default = None, max_length = 120)
    extractor_enabled: bool = True
    ocr_enabled: bool = True
    reranker_enabled: bool = False
    extractor_instance_id: str | None = Field(default = None, max_length = 120)
    ocr_instance_id: str | None = Field(default = None, max_length = 120)
    embedder_instance_id: str | None = Field(default = None, max_length = 120)
    reranker_instance_id: str | None = Field(default = None, max_length = 120)
    chunk_size: int = Field(default = 900, ge = 200, le = 4000)
    chunk_overlap: int = Field(default = 120, ge = 0, le = 1000)


class CreateRAGDatasetRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 120)
    source_kind: RAGSourceKind
    description: str | None = Field(default = None, max_length = 1000)


class AppendRAGDatasetTextRequest(BaseModel):
    document_name: str = Field(min_length = 1, max_length = 240)
    text: str = Field(min_length = 1)


class CreateRAGCollectionRequest(BaseModel):
    name: str = Field(min_length = 1, max_length = 120)
    connection_profile_id: str = Field(min_length = 1)
    ingestion_profile_id: str = Field(min_length = 1)
    remote_collection_name: str = Field(min_length = 1, max_length = 240)


class PublishRAGDatasetRequest(BaseModel):
    dataset_id: str = Field(min_length = 1)


class RAGDocumentUploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    status: RAGDocumentUploadStatus
    error: str | None = None
