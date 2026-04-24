# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import APIRouter, UploadFile, status

from core.rag.service import get_rag_service
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
    RAGConnectionProfileListResponse,
    RAGConnectionProfileSummary,
    RAGDatasetDetailResponse,
    RAGDatasetListResponse,
    RAGDatasetSummary,
    RAGDiagnosticsResponse,
    RAGDocumentUploadResponse,
    RAGIngestionProfileListResponse,
    RAGIngestionProfileSummary,
    RAGJobListResponse,
    RAGJobSummary,
    RAGModuleActionResponse,
    RAGModuleCatalogResponse,
    RAGModuleInstallationListResponse,
    RAGModuleInstanceListResponse,
    RAGModuleInstanceSummary,
    RAGOverviewResponse,
    RAGProvidersResponse,
)

router = APIRouter()


@router.get("/overview", response_model = RAGOverviewResponse)
def get_overview():
    return get_rag_service().get_overview()


@router.get("/providers", response_model = RAGProvidersResponse)
def get_providers():
    return get_rag_service().get_providers()


@router.get("/modules/catalog", response_model = RAGModuleCatalogResponse)
def get_module_catalog():
    return get_rag_service().get_module_catalog()


@router.get("/modules/installations", response_model = RAGModuleInstallationListResponse)
def list_module_installations():
    return get_rag_service().list_module_installations()


@router.post("/modules/{module_id}/check", response_model = RAGModuleActionResponse)
def check_module(module_id: str):
    return get_rag_service().check_module(module_id)


@router.post("/modules/{module_id}/install", response_model = RAGModuleActionResponse)
def install_module(module_id: str):
    return get_rag_service().install_module(module_id)


@router.post("/modules/{module_id}/install-local", response_model = RAGModuleActionResponse)
def install_local_module(module_id: str, payload: InstallRAGModuleRequest):
    return get_rag_service().install_local_module(module_id, payload)


@router.get("/modules/instances", response_model = RAGModuleInstanceListResponse)
def list_module_instances():
    return get_rag_service().list_module_instances()


@router.post(
    "/modules/instances",
    response_model = RAGModuleInstanceSummary,
    status_code = status.HTTP_201_CREATED,
)
def create_module_instance(payload: CreateRAGModuleInstanceRequest):
    return get_rag_service().create_module_instance(payload)


@router.post("/modules/instances/{instance_id}/test", response_model = RAGModuleActionResponse)
def test_module_instance(instance_id: str):
    return get_rag_service().test_module_instance(instance_id)


@router.get("/connection-profiles", response_model = RAGConnectionProfileListResponse)
def list_connection_profiles():
    return get_rag_service().list_connection_profiles()


@router.post(
    "/connection-profiles",
    response_model = RAGConnectionProfileSummary,
    status_code = status.HTTP_201_CREATED,
)
def create_connection_profile(payload: CreateRAGConnectionProfileRequest):
    return get_rag_service().create_connection_profile(payload)


@router.get("/ingestion-profiles", response_model = RAGIngestionProfileListResponse)
def list_ingestion_profiles():
    return get_rag_service().list_ingestion_profiles()


@router.post(
    "/ingestion-profiles",
    response_model = RAGIngestionProfileSummary,
    status_code = status.HTTP_201_CREATED,
)
def create_ingestion_profile(payload: CreateRAGIngestionProfileRequest):
    return get_rag_service().create_ingestion_profile(payload)


@router.get("/datasets", response_model = RAGDatasetListResponse)
def list_datasets():
    return get_rag_service().list_datasets()


@router.post(
    "/datasets",
    response_model = RAGDatasetSummary,
    status_code = status.HTTP_201_CREATED,
)
def create_dataset(payload: CreateRAGDatasetRequest):
    return get_rag_service().create_dataset(payload)


@router.get("/datasets/{dataset_id}", response_model = RAGDatasetDetailResponse)
def get_dataset(dataset_id: str):
    return get_rag_service().get_dataset_detail(dataset_id)


@router.post(
    "/datasets/{dataset_id}/text",
    response_model = RAGDocumentUploadResponse,
    status_code = status.HTTP_201_CREATED,
)
def append_dataset_text(dataset_id: str, payload: AppendRAGDatasetTextRequest):
    document = get_rag_service().append_dataset_text(dataset_id, payload)
    return RAGDocumentUploadResponse(
        file_id = document.id,
        filename = document.document_name,
        size_bytes = document.size_bytes,
        status = "ok",
    )


@router.post(
    "/datasets/{dataset_id}/documents",
    response_model = RAGDocumentUploadResponse,
    status_code = status.HTTP_201_CREATED,
)
async def upload_dataset_document(dataset_id: str, file: UploadFile):
    return await get_rag_service().upload_dataset_document(dataset_id, file)


@router.get("/collections", response_model = RAGCollectionListResponse)
def list_collections():
    return get_rag_service().list_collections()


@router.post(
    "/collections",
    response_model = RAGCollectionSummary,
    status_code = status.HTTP_201_CREATED,
)
def create_collection(payload: CreateRAGCollectionRequest):
    return get_rag_service().create_collection(payload)


@router.post(
    "/collections/{collection_id}/publish",
    response_model = RAGJobSummary,
    status_code = status.HTTP_201_CREATED,
)
def publish_dataset(collection_id: str, payload: PublishRAGDatasetRequest):
    return get_rag_service().publish_dataset(collection_id, payload)


@router.post(
    "/collections/{collection_id}/reindex",
    response_model = RAGJobSummary,
    status_code = status.HTTP_201_CREATED,
)
def reindex_collection(collection_id: str):
    return get_rag_service().reindex_collection(collection_id)


@router.post(
    "/collections/{collection_id}/sync-open-webui",
    response_model = RAGJobSummary,
    status_code = status.HTTP_201_CREATED,
)
def sync_collection(collection_id: str):
    return get_rag_service().sync_open_webui(collection_id)


@router.get("/jobs", response_model = RAGJobListResponse)
def list_jobs():
    return get_rag_service().list_jobs()


@router.get("/diagnostics", response_model = RAGDiagnosticsResponse)
def get_diagnostics():
    return get_rag_service().get_diagnostics()
