// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

function stringifyErrorDetail(detail: unknown): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item) => stringifyErrorDetail(item))
      .filter((item) => item.length > 0);
    return parts.join("; ");
  }
  if (detail && typeof detail === "object") {
    const record = detail as Record<string, unknown>;
    const fieldPath = Array.isArray(record.loc)
      ? record.loc
          .map((item) => String(item))
          .filter((item) => item !== "body")
          .join(".")
      : "";
    const message = typeof record.msg === "string"
      ? record.msg
      : typeof record.message === "string"
        ? record.message
        : "";
    if (fieldPath && message) {
      return `${fieldPath}: ${message}`;
    }
    if (message) {
      return message;
    }
    try {
      return JSON.stringify(record);
    } catch {
      return String(record);
    }
  }
  return "";
}

async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown; message?: unknown };
    const detail = stringifyErrorDetail(payload.detail);
    if (detail) {
      return detail;
    }
    const message = stringifyErrorDetail(payload.message);
    if (message) {
      return message;
    }
    return `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readError(response));
  }
  return (await response.json()) as T;
}

export type RagBackendId = "qdrant" | "pgvector";
export type RagProviderStage = "first-class" | "design-only" | "future";
export type RagCollectionSyncStatus = "not-synced" | "pending" | "synced" | "error";
export type RagJobType = "publish" | "reindex" | "sync-open-webui";
export type RagJobStatus = "queued" | "running" | "completed" | "error";
export type RagSourceKind = "documents" | "normalized-text";
export type RagDatasetStatus = "empty" | "ready" | "processing" | "error";
export type RagConnectionStatus = "healthy" | "error" | "disabled" | "unknown";
export type RagModuleKind = "extractor" | "ocr" | "embedder" | "reranker" | "chunker" | "vector_store";
export type RagModuleSourceType =
  | "builtin"
  | "python_package"
  | "system_binary"
  | "local_model_path"
  | "service_url"
  | "wheel"
  | "wheelhouse";
export type RagModuleStatus = "available" | "missing" | "configured" | "error" | "disabled";

export interface RagProviderOption {
  id: string;
  label: string;
  description: string;
  enabled: boolean;
  stage: RagProviderStage;
  kind?: RagModuleKind | null;
  source_type?: RagModuleSourceType | null;
  package_name?: string | null;
  module_name?: string | null;
  binary_name?: string | null;
  default_model_id?: string | null;
  installable?: boolean;
  configurable?: boolean;
}

export interface RagProvidersResponse {
  vector_stores: RagProviderOption[];
  extractors: RagProviderOption[];
  ocr_engines: RagProviderOption[];
  embedders: RagProviderOption[];
  rerankers: RagProviderOption[];
}

export interface RagOverviewBackend {
  id: RagBackendId;
  enabled: boolean;
  stage: RagProviderStage;
}

export interface RagOverviewResponse {
  collections_total: number;
  jobs_total: number;
  datasets_total: number;
  documents_total: number;
  chunks_total: number;
  synced_collections_total: number;
  supported_backends: RagOverviewBackend[];
}

export interface RagModuleCatalogItem {
  id: string;
  kind: RagModuleKind;
  label: string;
  description: string;
  enabled: boolean;
  stage: RagProviderStage;
  source_type: RagModuleSourceType;
  package_name: string | null;
  module_name: string | null;
  binary_name: string | null;
  default_model_id: string | null;
  installable: boolean;
  configurable: boolean;
  dependencies: string[];
  config_schema: Record<string, unknown>;
}

export interface RagModuleCatalogResponse {
  items: RagModuleCatalogItem[];
}

export interface RagModuleInstallationSummary {
  module_id: string;
  kind: RagModuleKind;
  status: RagModuleStatus;
  source_type: RagModuleSourceType;
  version: string | null;
  path: string | null;
  last_checked_at: string | null;
  last_error: string | null;
  install_command: string | null;
}

export interface RagModuleInstallationListResponse {
  items: RagModuleInstallationSummary[];
}

export interface RagModuleInstanceSummary {
  id: string;
  name: string;
  module_id: string;
  kind: RagModuleKind;
  source_type: RagModuleSourceType;
  model_id: string | null;
  local_path: string | null;
  service_url: string | null;
  binary_path: string | null;
  enabled: boolean;
  status: RagModuleStatus;
  last_checked_at: string | null;
  last_error: string | null;
  created_at: string;
}

export interface RagModuleInstanceListResponse {
  items: RagModuleInstanceSummary[];
}

export interface CreateRagModuleInstanceParams {
  name: string;
  module_id: string;
  source_type: RagModuleSourceType;
  model_id?: string | null;
  local_path?: string | null;
  service_url?: string | null;
  binary_path?: string | null;
  enabled?: boolean;
}

export interface InstallRagModuleParams {
  package_path?: string | null;
  wheelhouse_path?: string | null;
}

export interface RagModuleActionResponse {
  module_id: string;
  status: RagModuleStatus;
  message: string;
  details: Record<string, unknown>;
}

export interface RagConnectionProfileSummary {
  id: string;
  name: string;
  backend: RagBackendId;
  base_url: string;
  default_collection_prefix: string | null;
  enabled: boolean;
  status: RagConnectionStatus;
  created_at: string;
}

export interface RagConnectionProfileListResponse {
  items: RagConnectionProfileSummary[];
}

export interface RagIngestionProfileSummary {
  id: string;
  name: string;
  extractor: string;
  ocr_engine: string;
  embedder: string;
  reranker: string | null;
  extractor_enabled: boolean;
  ocr_enabled: boolean;
  reranker_enabled: boolean;
  extractor_instance_id: string | null;
  ocr_instance_id: string | null;
  embedder_instance_id: string | null;
  reranker_instance_id: string | null;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
}

export interface RagIngestionProfileListResponse {
  items: RagIngestionProfileSummary[];
}

export interface RagDatasetDocumentSummary {
  id: string;
  dataset_id: string;
  document_name: string;
  mime_type: string;
  source_kind: RagSourceKind;
  size_bytes: number;
  text_char_count: number;
  content_hash: string;
  created_at: string;
  processing_status: "processing" | "ready" | "error";
  processing_error: string | null;
  processed_at: string | null;
  extractor: string | null;
  ocr_engine: string | null;
}

export interface RagDatasetSummary {
  id: string;
  name: string;
  source_kind: RagSourceKind;
  description: string | null;
  status: RagDatasetStatus;
  documents_count: number;
  chunks_count: number;
  total_characters: number;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface RagDatasetListResponse {
  items: RagDatasetSummary[];
}

export interface RagDatasetDetailResponse {
  dataset: RagDatasetSummary;
  documents: RagDatasetDocumentSummary[];
}

export interface RagIndexProjectionSummary {
  id: string;
  backend: RagBackendId;
  embedder: string;
  embedding_model: string;
  physical_collection_name: string;
  extractor: string;
  ocr_engine: string;
  chunk_recipe: string;
  status: string;
  version: number;
  source_document_count: number;
  indexed_at: string | null;
}

export interface RagCollectionSummary {
  id: string;
  name: string;
  backend: RagBackendId;
  connection_profile_id: string;
  connection_profile_name: string;
  ingestion_profile_id: string;
  ingestion_profile_name: string;
  remote_collection_name: string;
  documents_count: number;
  chunks_count: number;
  sync_status: RagCollectionSyncStatus;
  last_job_status: RagJobStatus | null;
  last_error: string | null;
  last_reindex_at: string | null;
  active_projection: RagIndexProjectionSummary;
}

export interface RagCollectionListResponse {
  items: RagCollectionSummary[];
}

export interface RagJobSummary {
  id: string;
  job_type: RagJobType;
  status: RagJobStatus;
  collection_id: string;
  collection_name: string;
  dataset_id: string | null;
  dataset_name: string | null;
  initiator: string;
  stage_label: string;
  progress_percent: number;
  duration_seconds: number | null;
  warning_count: number;
  created_at: string;
  finished_at: string | null;
}

export interface RagJobListResponse {
  items: RagJobSummary[];
}

export interface RagConnectionDiagnostic {
  id: string;
  name: string;
  backend: RagBackendId;
  status: RagConnectionStatus;
  details: Record<string, unknown>;
}

export interface RagDiagnosticsResponse {
  connection_profiles: RagConnectionDiagnostic[];
  collections: RagCollectionSummary[];
}

export interface RagCollectionInspectVectorState {
  backend: RagBackendId;
  status: string;
  physical_collection_name: string;
  details: Record<string, unknown>;
  collection_info: Record<string, unknown>;
}

export interface RagCollectionChunkStats {
  chunks_total: number;
  qdrant_points_total: number | null;
  documents_total: number;
  average_chunk_chars: number;
  min_chunk_chars: number;
  max_chunk_chars: number;
  missing_text_count: number;
  missing_indexed_at_count: number;
  inspected_limit: number;
  truncated: boolean;
}

export interface RagCollectionDistributionItem {
  id: string;
  label: string;
  count: number;
  value: number | null;
}

export interface RagCollectionInspectDistributions {
  documents: RagCollectionDistributionItem[];
  chunk_sizes: RagCollectionDistributionItem[];
  indexing_statuses: RagCollectionDistributionItem[];
}

export interface RagCollectionChunkPayload {
  point_id: string;
  text: string;
  file_id: string | null;
  document_id: string | null;
  source: string | null;
  hash: string | null;
  extractor: string | null;
  ocr_engine: string | null;
  embedding_config: Record<string, unknown>;
  chunk_recipe: string | null;
  indexed_at: string | null;
  chunk_index: number | null;
  payload: Record<string, unknown>;
}

export interface RagCollectionSearchResult extends RagCollectionChunkPayload {
  score: number;
}

export interface RagCollectionInspectResponse {
  collection: RagCollectionSummary;
  connection_profile: RagConnectionProfileSummary;
  ingestion_profile: RagIngestionProfileSummary;
  active_projection: RagIndexProjectionSummary;
  qdrant: RagCollectionInspectVectorState;
  stats: RagCollectionChunkStats;
  distributions: RagCollectionInspectDistributions;
  warnings: string[];
}

export interface RagCollectionSampleChunksResponse {
  collection_id: string;
  limit: number;
  offset: number;
  next_offset: number | null;
  items: RagCollectionChunkPayload[];
}

export interface SearchRagCollectionParams {
  query: string;
  limit: number;
}

export interface RagCollectionSearchResponse {
  collection_id: string;
  query: string;
  limit: number;
  embedding_model: string;
  results: RagCollectionSearchResult[];
}

export interface CreateRagConnectionProfileParams {
  name: string;
  backend: RagBackendId;
  base_url: string;
  api_key?: string | null;
  default_collection_prefix?: string | null;
  enabled?: boolean;
}

export interface CreateRagIngestionProfileParams {
  name: string;
  extractor: string;
  ocr_engine: string;
  embedder: string;
  reranker?: string | null;
  extractor_enabled?: boolean;
  ocr_enabled?: boolean;
  reranker_enabled?: boolean;
  extractor_instance_id?: string | null;
  ocr_instance_id?: string | null;
  embedder_instance_id?: string | null;
  reranker_instance_id?: string | null;
  chunk_size: number;
  chunk_overlap: number;
}

export interface CreateRagDatasetParams {
  name: string;
  source_kind: RagSourceKind;
  description?: string | null;
}

export interface AppendRagDatasetTextParams {
  document_name: string;
  text: string;
}

export interface CreateRagCollectionParams {
  name: string;
  connection_profile_id: string;
  ingestion_profile_id: string;
  remote_collection_name: string;
}

export interface PublishRagDatasetParams {
  dataset_id: string;
}

export interface RagDocumentUploadItem {
  file_id: string;
  filename: string;
  size_bytes: number;
  status: "ok" | "error";
  error: string | null;
}

export interface RagDocumentUploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  status: "ok" | "error";
  error: string | null;
  files: RagDocumentUploadItem[];
}

function uploadFileName(file: File): string {
  return (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name;
}

export async function fetchRagOverview(): Promise<RagOverviewResponse> {
  return parseJson<RagOverviewResponse>(await authFetch("/api/rag/overview"));
}

export async function fetchRagProviders(): Promise<RagProvidersResponse> {
  return parseJson<RagProvidersResponse>(await authFetch("/api/rag/providers"));
}

export async function fetchRagModuleCatalog(): Promise<RagModuleCatalogResponse> {
  return parseJson<RagModuleCatalogResponse>(await authFetch("/api/rag/modules/catalog"));
}

export async function fetchRagModuleInstallations(): Promise<RagModuleInstallationListResponse> {
  return parseJson<RagModuleInstallationListResponse>(
    await authFetch("/api/rag/modules/installations"),
  );
}

export async function checkRagModule(moduleId: string): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/${moduleId}/check`, {
      method: "POST",
    }),
  );
}

export async function installRagModule(moduleId: string): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/${moduleId}/install`, {
      method: "POST",
    }),
  );
}

export async function installLocalRagModule(
  moduleId: string,
  params: InstallRagModuleParams,
): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/${moduleId}/install-local`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function uninstallRagModulePackage(moduleId: string): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/${moduleId}/package`, {
      method: "DELETE",
    }),
  );
}

export async function fetchRagModuleInstances(): Promise<RagModuleInstanceListResponse> {
  return parseJson<RagModuleInstanceListResponse>(
    await authFetch("/api/rag/modules/instances"),
  );
}

export async function createRagModuleInstance(
  params: CreateRagModuleInstanceParams,
): Promise<RagModuleInstanceSummary> {
  return parseJson<RagModuleInstanceSummary>(
    await authFetch("/api/rag/modules/instances", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function deleteRagModuleInstance(instanceId: string): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/instances/${instanceId}`, {
      method: "DELETE",
    }),
  );
}

export async function testRagModuleInstance(instanceId: string): Promise<RagModuleActionResponse> {
  return parseJson<RagModuleActionResponse>(
    await authFetch(`/api/rag/modules/instances/${instanceId}/test`, {
      method: "POST",
    }),
  );
}

export async function fetchRagConnectionProfiles(): Promise<RagConnectionProfileListResponse> {
  return parseJson<RagConnectionProfileListResponse>(
    await authFetch("/api/rag/connection-profiles"),
  );
}

export async function createRagConnectionProfile(
  params: CreateRagConnectionProfileParams,
): Promise<RagConnectionProfileSummary> {
  return parseJson<RagConnectionProfileSummary>(
    await authFetch("/api/rag/connection-profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function fetchRagIngestionProfiles(): Promise<RagIngestionProfileListResponse> {
  return parseJson<RagIngestionProfileListResponse>(
    await authFetch("/api/rag/ingestion-profiles"),
  );
}

export async function createRagIngestionProfile(
  params: CreateRagIngestionProfileParams,
): Promise<RagIngestionProfileSummary> {
  return parseJson<RagIngestionProfileSummary>(
    await authFetch("/api/rag/ingestion-profiles", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function fetchRagDatasets(): Promise<RagDatasetListResponse> {
  return parseJson<RagDatasetListResponse>(await authFetch("/api/rag/datasets"));
}

export async function fetchRagDatasetDetail(datasetId: string): Promise<RagDatasetDetailResponse> {
  return parseJson<RagDatasetDetailResponse>(await authFetch(`/api/rag/datasets/${datasetId}`));
}

export async function createRagDataset(
  params: CreateRagDatasetParams,
): Promise<RagDatasetSummary> {
  return parseJson<RagDatasetSummary>(
    await authFetch("/api/rag/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function appendRagDatasetText(
  datasetId: string,
  params: AppendRagDatasetTextParams,
): Promise<RagDocumentUploadResponse> {
  return parseJson(
    await authFetch(`/api/rag/datasets/${datasetId}/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function uploadRagDatasetDocument(
  datasetId: string,
  file: File,
): Promise<RagDocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file, uploadFileName(file));
  return parseJson(
    await authFetch(`/api/rag/datasets/${datasetId}/documents`, {
      method: "POST",
      body: formData,
    }),
  );
}

export async function fetchRagCollections(): Promise<RagCollectionListResponse> {
  return parseJson<RagCollectionListResponse>(await authFetch("/api/rag/collections"));
}

export async function createRagCollection(
  params: CreateRagCollectionParams,
): Promise<RagCollectionSummary> {
  return parseJson<RagCollectionSummary>(
    await authFetch("/api/rag/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function fetchRagCollectionInspect(
  collectionId: string,
): Promise<RagCollectionInspectResponse> {
  return parseJson<RagCollectionInspectResponse>(
    await authFetch(`/api/rag/collections/${encodeURIComponent(collectionId)}/inspect`),
  );
}

export async function fetchRagCollectionSampleChunks(
  collectionId: string,
  params: { limit?: number; offset?: number } = {},
): Promise<RagCollectionSampleChunksResponse> {
  const searchParams = new URLSearchParams();
  if (params.limit !== undefined) {
    searchParams.set("limit", String(params.limit));
  }
  if (params.offset !== undefined) {
    searchParams.set("offset", String(params.offset));
  }
  const queryString = searchParams.toString();
  const suffix = queryString ? `?${queryString}` : "";
  return parseJson<RagCollectionSampleChunksResponse>(
    await authFetch(
      `/api/rag/collections/${encodeURIComponent(collectionId)}/sample-chunks${suffix}`,
    ),
  );
}

export async function searchRagCollection(
  collectionId: string,
  params: SearchRagCollectionParams,
): Promise<RagCollectionSearchResponse> {
  return parseJson<RagCollectionSearchResponse>(
    await authFetch(`/api/rag/collections/${encodeURIComponent(collectionId)}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function publishRagDatasetToCollection(
  collectionId: string,
  params: PublishRagDatasetParams,
): Promise<RagJobSummary> {
  return parseJson<RagJobSummary>(
    await authFetch(`/api/rag/collections/${collectionId}/publish`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    }),
  );
}

export async function reindexRagCollection(
  collectionId: string,
): Promise<RagJobSummary> {
  return parseJson<RagJobSummary>(
    await authFetch(`/api/rag/collections/${collectionId}/reindex`, {
      method: "POST",
    }),
  );
}

export async function syncRagCollectionToOpenWebUi(
  collectionId: string,
): Promise<RagJobSummary> {
  return parseJson<RagJobSummary>(
    await authFetch(`/api/rag/collections/${collectionId}/sync-open-webui`, {
      method: "POST",
    }),
  );
}

export async function fetchRagJobs(): Promise<RagJobListResponse> {
  return parseJson<RagJobListResponse>(await authFetch("/api/rag/jobs"));
}

export async function fetchRagDiagnostics(): Promise<RagDiagnosticsResponse> {
  return parseJson<RagDiagnosticsResponse>(await authFetch("/api/rag/diagnostics"));
}
