// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "@/components/ui/empty";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { toastError, toastSuccess } from "@/shared/toast";
import {
  RAGModuleCatalogBrowser,
  RAGModuleSelector,
  moduleStatusVariant,
} from "./components/rag-module-selector";
import {
  type RagCollectionSummary,
  type RagBackendId,
  type RagConnectionDiagnostic,
  type RagConnectionProfileSummary,
  type RagDatasetSummary,
  type RagIngestionProfileSummary,
  type RagJobStatus,
  type RagJobSummary,
  type RagModuleCatalogItem,
  type RagModuleInstallationSummary,
  type RagModuleInstanceSummary,
  type RagModuleKind,
  type RagModuleSourceType,
  type RagOverviewResponse,
  type RagProviderOption,
  appendRagDatasetText,
  checkRagModule,
  createRagCollection,
  createRagConnectionProfile,
  createRagDataset,
  createRagIngestionProfile,
  createRagModuleInstance,
  fetchRagCollections,
  fetchRagConnectionProfiles,
  fetchRagDatasets,
  fetchRagDiagnostics,
  fetchRagIngestionProfiles,
  fetchRagJobs,
  fetchRagModuleCatalog,
  fetchRagModuleInstallations,
  fetchRagModuleInstances,
  fetchRagOverview,
  fetchRagProviders,
  installLocalRagModule,
  installRagModule,
  publishRagDatasetToCollection,
  reindexRagCollection,
  syncRagCollectionToOpenWebUi,
  testRagModuleInstance,
  uploadRagDatasetDocument,
} from "./api/rag-api";
import {
  Book03Icon,
  CheckmarkCircle02Icon,
  Database02Icon,
  DocumentAttachmentIcon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useState, type ChangeEvent, type FormEvent } from "react";

type AdminRagTab = "configure" | "modules" | "datasets" | "collections" | "jobs";

type ConnectionFormState = {
  name: string;
  backend: RagBackendId;
  baseUrl: string;
  apiKey: string;
  prefix: string;
};

type IngestionFormState = {
  name: string;
  extractor: string;
  ocrEngine: string;
  embedder: string;
  reranker: string;
  extractorEnabled: boolean;
  ocrEnabled: boolean;
  rerankerEnabled: boolean;
  extractorInstanceId: string;
  ocrInstanceId: string;
  embedderInstanceId: string;
  rerankerInstanceId: string;
  chunkSize: string;
  chunkOverlap: string;
};

type ModuleFormState = {
  name: string;
  moduleId: string;
  sourceType: RagModuleSourceType;
  modelId: string;
  localPath: string;
  serviceUrl: string;
  binaryPath: string;
};

type LocalInstallFormState = {
  moduleId: string;
  packagePath: string;
  wheelhousePath: string;
};

type DatasetFormState = {
  name: string;
  sourceKind: "documents" | "normalized-text";
  description: string;
};

type AppendTextFormState = {
  datasetId: string;
  documentName: string;
  text: string;
};

type CollectionFormState = {
  name: string;
  connectionProfileId: string;
  ingestionProfileId: string;
  remoteCollectionName: string;
};

type PublishFormState = {
  collectionId: string;
  datasetId: string;
};

type ProfileModulePreview = {
  label: string;
  kind: RagModuleKind;
  name: string;
  target: string;
  status: RagModuleInstanceSummary["status"];
  warning: string | null;
};

const EMPTY_CONNECTION_FORM: ConnectionFormState = {
  name: "",
  backend: "qdrant",
  baseUrl: "",
  apiKey: "",
  prefix: "unsloth",
};

const EMPTY_INGESTION_FORM: IngestionFormState = {
  name: "",
  extractor: "",
  ocrEngine: "",
  embedder: "",
  reranker: "none",
  extractorEnabled: true,
  ocrEnabled: true,
  rerankerEnabled: false,
  extractorInstanceId: "",
  ocrInstanceId: "",
  embedderInstanceId: "",
  rerankerInstanceId: "",
  chunkSize: "900",
  chunkOverlap: "120",
};

const EMPTY_MODULE_FORM: ModuleFormState = {
  name: "",
  moduleId: "",
  sourceType: "local_model_path",
  modelId: "",
  localPath: "",
  serviceUrl: "",
  binaryPath: "",
};

const EMPTY_LOCAL_INSTALL_FORM: LocalInstallFormState = {
  moduleId: "",
  packagePath: "",
  wheelhousePath: "",
};

const EMPTY_DATASET_FORM: DatasetFormState = {
  name: "",
  sourceKind: "documents",
  description: "",
};

const EMPTY_APPEND_TEXT_FORM: AppendTextFormState = {
  datasetId: "",
  documentName: "",
  text: "",
};

const EMPTY_COLLECTION_FORM: CollectionFormState = {
  name: "",
  connectionProfileId: "",
  ingestionProfileId: "",
  remoteCollectionName: "",
};

const EMPTY_PUBLISH_FORM: PublishFormState = {
  collectionId: "",
  datasetId: "",
};

function metricLabel(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function jobStatusVariant(status: RagJobStatus): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "completed":
      return "default";
    case "running":
      return "secondary";
    case "error":
      return "destructive";
    default:
      return "outline";
  }
}

function syncStatusVariant(status: RagCollectionSummary["sync_status"]): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "synced":
      return "default";
    case "pending":
      return "secondary";
    case "error":
      return "destructive";
    default:
      return "outline";
  }
}

function connectionStatusVariant(status: RagConnectionDiagnostic["status"]): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "healthy":
      return "default";
    case "disabled":
      return "secondary";
    case "error":
      return "destructive";
    default:
      return "outline";
  }
}

function connectionStatusLabel(status: RagConnectionDiagnostic["status"]): string {
  switch (status) {
    case "unknown":
      return "unchecked";
    default:
      return status;
  }
}

function providerBadge(provider: RagProviderOption) {
  return (
    <Badge key={provider.id} variant={provider.enabled ? "outline" : "secondary"}>
      {provider.label}
      {!provider.enabled ? " · design-only" : ""}
    </Badge>
  );
}

function moduleInstanceTarget(instance: RagModuleInstanceSummary): string {
  return (
    instance.local_path
    || instance.model_id
    || instance.service_url
    || instance.binary_path
    || instance.module_id
  );
}

function moduleCatalogTarget(
  item: RagModuleCatalogItem | undefined,
  installation: RagModuleInstallationSummary | undefined,
): string {
  if (!item) {
    return "not configured";
  }
  return (
    installation?.path
    || item.default_model_id
    || item.package_name
    || item.binary_name
    || item.source_type
  );
}

export function AdminRagPage() {
  const [overview, setOverview] = useState<RagOverviewResponse | null>(null);
  const [providers, setProviders] = useState<Awaited<ReturnType<typeof fetchRagProviders>> | null>(null);
  const [moduleCatalog, setModuleCatalog] = useState<RagModuleCatalogItem[]>([]);
  const [moduleInstallations, setModuleInstallations] = useState<RagModuleInstallationSummary[]>([]);
  const [moduleInstances, setModuleInstances] = useState<RagModuleInstanceSummary[]>([]);
  const [connectionProfiles, setConnectionProfiles] = useState<RagConnectionProfileSummary[]>([]);
  const [ingestionProfiles, setIngestionProfiles] = useState<RagIngestionProfileSummary[]>([]);
  const [datasets, setDatasets] = useState<RagDatasetSummary[]>([]);
  const [collections, setCollections] = useState<RagCollectionSummary[]>([]);
  const [jobs, setJobs] = useState<RagJobSummary[]>([]);
  const [diagnostics, setDiagnostics] = useState<RagConnectionDiagnostic[]>([]);
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [connectionForm, setConnectionForm] = useState(EMPTY_CONNECTION_FORM);
  const [ingestionForm, setIngestionForm] = useState(EMPTY_INGESTION_FORM);
  const [moduleForm, setModuleForm] = useState(EMPTY_MODULE_FORM);
  const [localInstallForm, setLocalInstallForm] = useState(EMPTY_LOCAL_INSTALL_FORM);
  const [datasetForm, setDatasetForm] = useState(EMPTY_DATASET_FORM);
  const [appendTextForm, setAppendTextForm] = useState(EMPTY_APPEND_TEXT_FORM);
  const [collectionForm, setCollectionForm] = useState(EMPTY_COLLECTION_FORM);
  const [publishForm, setPublishForm] = useState(EMPTY_PUBLISH_FORM);
  const [uploadDatasetId, setUploadDatasetId] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<AdminRagTab>("configure");

  const reloadAll = useCallback(async () => {
    const [
      overviewData,
      providersData,
      moduleCatalogData,
      moduleInstallationsData,
      moduleInstancesData,
      connectionProfilesData,
      ingestionProfilesData,
      datasetsData,
      collectionsData,
      jobsData,
      diagnosticsData,
    ] = await Promise.all([
      fetchRagOverview(),
      fetchRagProviders(),
      fetchRagModuleCatalog(),
      fetchRagModuleInstallations(),
      fetchRagModuleInstances(),
      fetchRagConnectionProfiles(),
      fetchRagIngestionProfiles(),
      fetchRagDatasets(),
      fetchRagCollections(),
      fetchRagJobs(),
      fetchRagDiagnostics(),
    ]);

    setOverview(overviewData);
    setProviders(providersData);
    setModuleCatalog(moduleCatalogData.items);
    setModuleInstallations(moduleInstallationsData.items);
    setModuleInstances(moduleInstancesData.items);
    setConnectionProfiles(connectionProfilesData.items);
    setIngestionProfiles(ingestionProfilesData.items);
    setDatasets(datasetsData.items);
    setCollections(collectionsData.items);
    setJobs(jobsData.items);
    setDiagnostics(diagnosticsData.connection_profiles);
  }, []);

  useEffect(() => {
    let active = true;

    setLoading(true);
    reloadAll()
      .catch((error) => {
        if (!active) return;
        toastError(error instanceof Error ? error.message : "Failed to load RAG workspace.");
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [reloadAll]);

  useEffect(() => {
    if (!providers) {
      return;
    }
    setIngestionForm((current) => ({
      ...current,
      extractor: current.extractor || providers.extractors.find((item) => item.enabled)?.id || "",
      ocrEngine: current.ocrEngine || providers.ocr_engines.find((item) => item.enabled)?.id || "",
      embedder: current.embedder || providers.embedders.find((item) => item.enabled)?.id || "",
      reranker: current.reranker || providers.rerankers.find((item) => item.enabled)?.id || "none",
    }));
  }, [providers]);

  useEffect(() => {
    const configurableModule = moduleCatalog.find((item) => item.configurable && item.enabled);
    if (!configurableModule) {
      return;
    }
    setModuleForm((current) => ({
      ...current,
      moduleId: current.moduleId || configurableModule.id,
      sourceType: current.sourceType || configurableModule.source_type,
      modelId: current.modelId || configurableModule.default_model_id || "",
    }));
    setLocalInstallForm((current) => ({
      ...current,
      moduleId: current.moduleId || configurableModule.id,
    }));
  }, [moduleCatalog]);

  useEffect(() => {
    if (!appendTextForm.datasetId && datasets[0]?.id) {
      setAppendTextForm((current) => ({ ...current, datasetId: datasets[0]?.id || "" }));
    }
    if (!uploadDatasetId && datasets[0]?.id) {
      setUploadDatasetId(datasets[0]?.id || "");
    }
    if (!publishForm.datasetId && datasets[0]?.id) {
      setPublishForm((current) => ({ ...current, datasetId: datasets[0]?.id || "" }));
    }
  }, [appendTextForm.datasetId, datasets, publishForm.datasetId, uploadDatasetId]);

  useEffect(() => {
    if (!collectionForm.connectionProfileId && connectionProfiles[0]?.id) {
      setCollectionForm((current) => ({
        ...current,
        connectionProfileId: connectionProfiles[0]?.id || "",
      }));
    }
  }, [collectionForm.connectionProfileId, connectionProfiles]);

  useEffect(() => {
    if (!collectionForm.ingestionProfileId && ingestionProfiles[0]?.id) {
      setCollectionForm((current) => ({
        ...current,
        ingestionProfileId: ingestionProfiles[0]?.id || "",
      }));
    }
  }, [collectionForm.ingestionProfileId, ingestionProfiles]);

  useEffect(() => {
    if (!publishForm.collectionId && collections[0]?.id) {
      setPublishForm((current) => ({ ...current, collectionId: collections[0]?.id || "" }));
    }
  }, [collections, publishForm.collectionId]);

  const supportedBackends = useMemo(
    () => overview?.supported_backends.map((backend) => (
      <Badge key={backend.id} variant={backend.enabled ? "outline" : "secondary"}>
        {backend.id}
        {!backend.enabled ? " · design-only" : ""}
      </Badge>
    )) ?? [],
    [overview],
  );

  const availableVectorStores = useMemo(() => {
    if (providers?.vector_stores?.length) {
      return providers.vector_stores;
    }
    return [
      {
        id: "qdrant" as RagBackendId,
        label: "Qdrant",
        description: "Active vector store backend.",
        enabled: true,
        stage: "first-class" as const,
      },
    ];
  }, [providers]);

  const diagnosticsByProfileId = useMemo(
    () => new Map(diagnostics.map((item) => [item.id, item])),
    [diagnostics],
  );

  const moduleInstallationsById = useMemo(
    () => new Map(moduleInstallations.map((item) => [item.module_id, item])),
    [moduleInstallations],
  );

  const moduleCatalogById = useMemo(
    () => new Map(moduleCatalog.map((item) => [item.id, item])),
    [moduleCatalog],
  );

  const moduleInstancesById = useMemo(
    () => new Map(moduleInstances.map((item) => [item.id, item])),
    [moduleInstances],
  );

  const selectedModule = moduleCatalogById.get(moduleForm.moduleId) ?? null;

  const selectedCollectionIngestionProfile = useMemo(
    () => ingestionProfiles.find((item) => item.id === collectionForm.ingestionProfileId) ?? null,
    [collectionForm.ingestionProfileId, ingestionProfiles],
  );

  const selectedProfileModules = useMemo<ProfileModulePreview[]>(() => {
    if (!selectedCollectionIngestionProfile) {
      return [];
    }

    const resolve = ({
      label,
      kind,
      enabled,
      moduleId,
      instanceId,
    }: {
      label: string;
      kind: RagModuleKind;
      enabled: boolean;
      moduleId: string | null;
      instanceId: string | null;
    }): ProfileModulePreview => {
      if (!enabled) {
        return {
          label,
          kind,
          name: "Disabled",
          target: "This stage is disabled in the ingestion profile.",
          status: "disabled",
          warning: null,
        };
      }
      if (instanceId) {
        const instance = moduleInstancesById.get(instanceId);
        if (!instance) {
          return {
            label,
            kind,
            name: "Missing instance",
            target: instanceId,
            status: "error",
            warning: "The profile points to a module instance that no longer exists.",
          };
        }
        return {
          label,
          kind,
          name: instance.name,
          target: moduleInstanceTarget(instance),
          status: instance.status,
          warning: instance.status === "error" || instance.status === "missing"
            ? instance.last_error || "The selected module instance is not ready."
            : null,
        };
      }

      const catalogItem = moduleId
        ? moduleCatalogById.get(moduleId) ?? moduleCatalogById.get(`${moduleId}-reranker`)
        : undefined;
      const installation = moduleId ? moduleInstallationsById.get(moduleId) : undefined;
      const status = installation?.status ?? (catalogItem?.enabled ? "available" : "disabled");
      return {
        label,
        kind,
        name: catalogItem?.label ?? moduleId ?? "Not configured",
        target: moduleCatalogTarget(catalogItem, installation),
        status,
        warning: status === "error" || status === "missing"
          ? installation?.last_error || "The selected module is not available."
          : null,
      };
    };

    return [
      resolve({
        label: "Extractor",
        kind: "extractor",
        enabled: selectedCollectionIngestionProfile.extractor_enabled,
        moduleId: selectedCollectionIngestionProfile.extractor,
        instanceId: selectedCollectionIngestionProfile.extractor_instance_id,
      }),
      resolve({
        label: "OCR",
        kind: "ocr",
        enabled: selectedCollectionIngestionProfile.ocr_enabled,
        moduleId: selectedCollectionIngestionProfile.ocr_engine,
        instanceId: selectedCollectionIngestionProfile.ocr_instance_id,
      }),
      resolve({
        label: "Embedder",
        kind: "embedder",
        enabled: true,
        moduleId: selectedCollectionIngestionProfile.embedder,
        instanceId: selectedCollectionIngestionProfile.embedder_instance_id,
      }),
      resolve({
        label: "Reranker",
        kind: "reranker",
        enabled: selectedCollectionIngestionProfile.reranker_enabled && Boolean(selectedCollectionIngestionProfile.reranker),
        moduleId: selectedCollectionIngestionProfile.reranker,
        instanceId: selectedCollectionIngestionProfile.reranker_instance_id,
      }),
    ];
  }, [
    moduleCatalogById,
    moduleInstallationsById,
    moduleInstancesById,
    selectedCollectionIngestionProfile,
  ]);

  const withBusyAction = useCallback(
    async (key: string, action: () => Promise<void>) => {
      try {
        setBusyAction(key);
        await action();
      } catch (error) {
        toastError(error instanceof Error ? error.message : "RAG action failed.");
      } finally {
        setBusyAction(null);
      }
    },
    [],
  );

  const onCreateConnectionProfile = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!connectionForm.name.trim()) {
        toastError("Enter a connection profile name.");
        return;
      }
      if (!connectionForm.baseUrl.trim()) {
        toastError("Enter the base URL for the vector store.");
        return;
      }
      await withBusyAction("create-connection-profile", async () => {
        await createRagConnectionProfile({
          name: connectionForm.name.trim(),
          backend: connectionForm.backend,
          base_url: connectionForm.baseUrl.trim(),
          api_key: connectionForm.apiKey.trim() || null,
          default_collection_prefix: connectionForm.prefix.trim() || null,
          enabled: true,
        });
        setConnectionForm(EMPTY_CONNECTION_FORM);
        await reloadAll();
        toastSuccess("Connection profile created.");
      });
    },
    [connectionForm, reloadAll, withBusyAction],
  );

  const onCreateIngestionProfile = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!ingestionForm.name.trim()) {
        toastError("Enter an ingestion profile name.");
        return;
      }
      if (
        (ingestionForm.extractorEnabled && !ingestionForm.extractor)
        || (ingestionForm.ocrEnabled && !ingestionForm.ocrEngine)
        || !ingestionForm.embedder
      ) {
        toastError("Choose active extraction components and embedder.");
        return;
      }
      const chunkSize = Number(ingestionForm.chunkSize);
      const chunkOverlap = Number(ingestionForm.chunkOverlap);
      if (!Number.isFinite(chunkSize) || !Number.isFinite(chunkOverlap)) {
        toastError("Chunk size and overlap must be numeric.");
        return;
      }
      await withBusyAction("create-ingestion-profile", async () => {
        await createRagIngestionProfile({
          name: ingestionForm.name.trim(),
          extractor: ingestionForm.extractor,
          ocr_engine: ingestionForm.ocrEngine,
          embedder: ingestionForm.embedder,
          reranker: ingestionForm.reranker === "none" ? null : ingestionForm.reranker,
          extractor_enabled: ingestionForm.extractorEnabled,
          ocr_enabled: ingestionForm.ocrEnabled,
          reranker_enabled: ingestionForm.rerankerEnabled,
          extractor_instance_id: ingestionForm.extractorInstanceId || null,
          ocr_instance_id: ingestionForm.ocrInstanceId || null,
          embedder_instance_id: ingestionForm.embedderInstanceId || null,
          reranker_instance_id: ingestionForm.rerankerInstanceId || null,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        });
        setIngestionForm((current) => ({ ...EMPTY_INGESTION_FORM, ...current, name: "" }));
        await reloadAll();
        toastSuccess("Ingestion profile created.");
      });
    },
    [ingestionForm, reloadAll, withBusyAction],
  );

  const onCheckModule = useCallback(
    async (moduleId: string) => {
      await withBusyAction(`module-check:${moduleId}`, async () => {
        const response = await checkRagModule(moduleId);
        await reloadAll();
        toastSuccess(`Module check: ${response.status}.`);
      });
    },
    [reloadAll, withBusyAction],
  );

  const onInstallModule = useCallback(
    async (moduleId: string) => {
      await withBusyAction(`module-install:${moduleId}`, async () => {
        const response = await installRagModule(moduleId);
        await reloadAll();
        toastSuccess(`Module install: ${response.status}.`);
      });
    },
    [reloadAll, withBusyAction],
  );

  const onInstallLocalModule = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!localInstallForm.moduleId) {
        toastError("Choose a module first.");
        return;
      }
      if (!localInstallForm.packagePath.trim() && !localInstallForm.wheelhousePath.trim()) {
        toastError("Set a wheel file path or a wheelhouse path.");
        return;
      }
      await withBusyAction("module-install-local", async () => {
        const response = await installLocalRagModule(localInstallForm.moduleId, {
          package_path: localInstallForm.packagePath.trim() || null,
          wheelhouse_path: localInstallForm.wheelhousePath.trim() || null,
        });
        setLocalInstallForm((current) => ({ ...current, packagePath: "", wheelhousePath: "" }));
        await reloadAll();
        toastSuccess(`Local install: ${response.status}.`);
      });
    },
    [localInstallForm, reloadAll, withBusyAction],
  );

  const onCreateModuleInstance = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!moduleForm.name.trim()) {
        toastError("Enter a module instance name.");
        return;
      }
      if (!moduleForm.moduleId) {
        toastError("Choose a module.");
        return;
      }
      if (moduleForm.sourceType === "local_model_path" && !moduleForm.localPath.trim() && !moduleForm.modelId.trim()) {
        toastError("Set a local model path or model ID.");
        return;
      }
      if (moduleForm.sourceType === "service_url" && !moduleForm.serviceUrl.trim()) {
        toastError("Set a service URL.");
        return;
      }
      if (moduleForm.sourceType === "system_binary" && !moduleForm.binaryPath.trim()) {
        toastError("Set a binary path or command name.");
        return;
      }
      await withBusyAction("create-module-instance", async () => {
        await createRagModuleInstance({
          name: moduleForm.name.trim(),
          module_id: moduleForm.moduleId,
          source_type: moduleForm.sourceType,
          model_id: moduleForm.modelId.trim() || null,
          local_path: moduleForm.localPath.trim() || null,
          service_url: moduleForm.serviceUrl.trim() || null,
          binary_path: moduleForm.binaryPath.trim() || null,
          enabled: true,
        });
        setModuleForm((current) => ({
          ...EMPTY_MODULE_FORM,
          moduleId: current.moduleId,
          sourceType: current.sourceType,
        }));
        await reloadAll();
        toastSuccess("Module instance created.");
      });
    },
    [moduleForm, reloadAll, withBusyAction],
  );

  const onTestModuleInstance = useCallback(
    async (instanceId: string) => {
      await withBusyAction(`module-instance-test:${instanceId}`, async () => {
        const response = await testRagModuleInstance(instanceId);
        await reloadAll();
        toastSuccess(`Module instance test: ${response.status}.`);
      });
    },
    [reloadAll, withBusyAction],
  );

  const onCreateDataset = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!datasetForm.name.trim()) {
        toastError("Enter a dataset name.");
        return;
      }
      await withBusyAction("create-dataset", async () => {
        await createRagDataset({
          name: datasetForm.name.trim(),
          source_kind: datasetForm.sourceKind,
          description: datasetForm.description.trim() || null,
        });
        setDatasetForm(EMPTY_DATASET_FORM);
        await reloadAll();
        toastSuccess("RAG dataset created.");
      });
    },
    [datasetForm, reloadAll, withBusyAction],
  );

  const onAppendText = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!appendTextForm.datasetId) {
        toastError("Choose a dataset first.");
        return;
      }
      if (!appendTextForm.documentName.trim()) {
        toastError("Enter a document name.");
        return;
      }
      if (!appendTextForm.text.trim()) {
        toastError("Paste normalized text before submitting.");
        return;
      }
      await withBusyAction("append-text", async () => {
        await appendRagDatasetText(appendTextForm.datasetId, {
          document_name: appendTextForm.documentName.trim(),
          text: appendTextForm.text,
        });
        setAppendTextForm((current) => ({ ...current, documentName: "", text: "" }));
        await reloadAll();
        toastSuccess("Normalized text added to dataset.");
      });
    },
    [appendTextForm, reloadAll, withBusyAction],
  );

  const onUploadDocument = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!uploadFile || !uploadDatasetId) {
        toastError("Choose a dataset and a file first.");
        return;
      }
      await withBusyAction("upload-document", async () => {
        await uploadRagDatasetDocument(uploadDatasetId, uploadFile);
        setUploadFile(null);
        await reloadAll();
        toastSuccess("Document uploaded to dataset.");
      });
    },
    [reloadAll, uploadDatasetId, uploadFile, withBusyAction],
  );

  const onCreateCollection = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!collectionForm.name.trim()) {
        toastError("Enter a collection name.");
        return;
      }
      if (!collectionForm.connectionProfileId || !collectionForm.ingestionProfileId) {
        toastError("Choose both a connection profile and an ingestion profile.");
        return;
      }
      if (!collectionForm.remoteCollectionName.trim()) {
        toastError("Enter the remote collection name.");
        return;
      }
      await withBusyAction("create-collection", async () => {
        await createRagCollection({
          name: collectionForm.name.trim(),
          connection_profile_id: collectionForm.connectionProfileId,
          ingestion_profile_id: collectionForm.ingestionProfileId,
          remote_collection_name: collectionForm.remoteCollectionName.trim(),
        });
        setCollectionForm((current) => ({
          ...EMPTY_COLLECTION_FORM,
          connectionProfileId: current.connectionProfileId,
          ingestionProfileId: current.ingestionProfileId,
        }));
        await reloadAll();
        toastSuccess("Collection created.");
      });
    },
    [collectionForm, reloadAll, withBusyAction],
  );

  const onPublishDataset = useCallback(
    async (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      if (!publishForm.collectionId || !publishForm.datasetId) {
        toastError("Choose both a collection and a dataset.");
        return;
      }
      await withBusyAction("publish-dataset", async () => {
        await publishRagDatasetToCollection(publishForm.collectionId, {
          dataset_id: publishForm.datasetId,
        });
        await reloadAll();
        toastSuccess("Dataset published to collection.");
      });
    },
    [publishForm, reloadAll, withBusyAction],
  );

  const onReindex = useCallback(
    async (collectionId: string) => {
      await withBusyAction(`reindex:${collectionId}`, async () => {
        await reindexRagCollection(collectionId);
        await reloadAll();
        toastSuccess("Collection reindexed.");
      });
    },
    [reloadAll, withBusyAction],
  );

  const onSync = useCallback(
    async (collectionId: string) => {
      await withBusyAction(`sync:${collectionId}`, async () => {
        await syncRagCollectionToOpenWebUi(collectionId);
        await reloadAll();
        toastSuccess("Collection marked as synced for Open WebUI.");
      });
    },
    [reloadAll, withBusyAction],
  );

  const onFileChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    setUploadFile(event.target.files?.[0] ?? null);
  }, []);

  if (loading && !overview) {
    return (
      <div className="flex min-h-[40vh] items-center justify-center">
        <Spinner className="size-8" />
      </div>
    );
  }

  const metrics = overview
    ? [
        { label: "Collections", value: overview.collections_total },
        { label: "Datasets", value: overview.datasets_total },
        { label: "Documents", value: overview.documents_total },
        { label: "Chunks", value: overview.chunks_total },
        { label: "Jobs", value: overview.jobs_total },
      ]
    : [];

  const healthyConnections = diagnostics.filter((item) => item.status === "healthy").length;
  const runningJobs = jobs.filter((item) => item.status === "running").length;
  const queuedJobs = jobs.filter((item) => item.status === "queued").length;
  const erroredJobs = jobs.filter((item) => item.status === "error").length;

  const subtitle =
    activeTab === "datasets"
      ? "Create corpora and add source material for ingestion."
      : activeTab === "modules"
        ? "Check, install and bind modular RAG components."
      : activeTab === "collections"
        ? "Publish datasets, reindex projections and sync knowledge targets."
        : activeTab === "jobs"
          ? "Inspect publish, reindex and Open WebUI sync activity."
          : "Configure vector targets and ingestion defaults.";

  const renderConfigureTab = () => (
    <div className="flex min-w-0 flex-col gap-4 md:gap-6">
      {providers ? (
        <SectionCard
          title="Available Providers"
          description="Current extraction, OCR and embedding options available to the RAG workspace."
          icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
          accent="blue"
        >
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl border border-border/70 bg-background/80 p-4">
              <div className="text-xs font-medium text-muted-foreground">Extractors / OCR</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {providers.extractors.map(providerBadge)}
                {providers.ocr_engines.map(providerBadge)}
              </div>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 p-4">
              <div className="text-xs font-medium text-muted-foreground">Embedders / Rerankers</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {providers.embedders.map(providerBadge)}
                {providers.rerankers.map(providerBadge)}
              </div>
            </div>
          </div>
        </SectionCard>
      ) : null}

      <div className="grid gap-4 md:gap-6 xl:grid-cols-2">
        <SectionCard
          title="Connection Profiles"
          description="External vector targets. `Qdrant` is active now; other backends stay behind the adapter layer."
          icon={<HugeiconsIcon icon={Search01Icon} className="size-5" />}
          badge={connectionProfiles.length > 0 ? String(connectionProfiles.length) : undefined}
        >
          <form className="space-y-4" onSubmit={onCreateConnectionProfile}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-connection-name">Profile name</Label>
                <Input
                  id="rag-connection-name"
                  value={connectionForm.name}
                  onChange={(event) => setConnectionForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Primary Qdrant"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-connection-backend">Backend</Label>
                <Select
                  value={connectionForm.backend}
                  onValueChange={(value: RagBackendId) =>
                    setConnectionForm((current) => ({ ...current, backend: value }))
                  }
                >
                  <SelectTrigger id="rag-connection-backend">
                    <SelectValue placeholder="Choose backend" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableVectorStores.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id} disabled={!provider.enabled}>
                        {provider.label}
                        {!provider.enabled ? " · design-only" : ""}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-connection-url">Base URL</Label>
              <Input
                id="rag-connection-url"
                value={connectionForm.baseUrl}
                onChange={(event) => setConnectionForm((current) => ({ ...current, baseUrl: event.target.value }))}
                placeholder="http://127.0.0.1:6333"
              />
              <div className="text-xs text-muted-foreground">
                For a local setup use `http://127.0.0.1:6333`.
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-connection-prefix">Collection prefix</Label>
              <Input
                id="rag-connection-prefix"
                value={connectionForm.prefix}
                onChange={(event) => setConnectionForm((current) => ({ ...current, prefix: event.target.value }))}
                placeholder="unsloth"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-connection-api-key">API key</Label>
              <Input
                id="rag-connection-api-key"
                type="password"
                value={connectionForm.apiKey}
                onChange={(event) => setConnectionForm((current) => ({ ...current, apiKey: event.target.value }))}
                placeholder="Optional"
              />
            </div>
            <Button type="submit" disabled={busyAction === "create-connection-profile"}>
              {busyAction === "create-connection-profile" ? <Spinner className="size-4" /> : "Create Connection Profile"}
            </Button>
          </form>

          <Separator className="my-4" />

          <div className="space-y-3">
            {connectionProfiles.length > 0 ? connectionProfiles.map((item) => {
              const diagnostic = diagnosticsByProfileId.get(item.id);
              const effectiveStatus = diagnostic?.status ?? item.status;
              return (
                <div key={item.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="text-sm font-semibold">{item.name}</div>
                      <div className="text-xs text-muted-foreground">{item.base_url}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{item.backend}</Badge>
                      <Badge variant={connectionStatusVariant(effectiveStatus)}>
                        {connectionStatusLabel(effectiveStatus)}
                      </Badge>
                    </div>
                  </div>
                </div>
              );
            }) : (
              <Empty>
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <HugeiconsIcon icon={Database02Icon} className="size-5" />
                  </EmptyMedia>
                  <EmptyTitle>No connection profiles yet</EmptyTitle>
                  <EmptyDescription>Create the first `Qdrant` connection profile to start publishing corpora.</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </div>
        </SectionCard>

        <SectionCard
          title="Ingestion Profiles"
          description="Extraction, OCR and chunk defaults applied when datasets are published into a collection."
          icon={<HugeiconsIcon icon={Book03Icon} className="size-5" />}
          badge={ingestionProfiles.length > 0 ? String(ingestionProfiles.length) : undefined}
        >
          <form className="space-y-4" onSubmit={onCreateIngestionProfile}>
            <div className="space-y-2">
              <Label htmlFor="rag-ingestion-name">Profile name</Label>
              <Input
                id="rag-ingestion-name"
                value={ingestionForm.name}
                onChange={(event) => setIngestionForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Docs Default"
              />
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="flex items-center justify-between gap-3 rounded-2xl border border-border/70 bg-background/80 p-3 text-sm">
                <span>
                  <span className="block font-medium">Extractor</span>
                  <span className="block text-xs text-muted-foreground">Use document extraction</span>
                </span>
                <Switch
                  checked={ingestionForm.extractorEnabled}
                  onCheckedChange={(checked) =>
                    setIngestionForm((current) => ({ ...current, extractorEnabled: checked }))
                  }
                />
              </label>
              <label className="flex items-center justify-between gap-3 rounded-2xl border border-border/70 bg-background/80 p-3 text-sm">
                <span>
                  <span className="block font-medium">OCR</span>
                  <span className="block text-xs text-muted-foreground">Enable scanned text path</span>
                </span>
                <Switch
                  checked={ingestionForm.ocrEnabled}
                  onCheckedChange={(checked) =>
                    setIngestionForm((current) => ({ ...current, ocrEnabled: checked }))
                  }
                />
              </label>
              <label className="flex items-center justify-between gap-3 rounded-2xl border border-border/70 bg-background/80 p-3 text-sm">
                <span>
                  <span className="block font-medium">Reranker</span>
                  <span className="block text-xs text-muted-foreground">Optional second pass</span>
                </span>
                <Switch
                  checked={ingestionForm.rerankerEnabled}
                  onCheckedChange={(checked) =>
                    setIngestionForm((current) => ({ ...current, rerankerEnabled: checked }))
                  }
                />
              </label>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-extractor">Extractor</Label>
                <Select
                  value={ingestionForm.extractor}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, extractor: value }))}
                  disabled={!ingestionForm.extractorEnabled}
                >
                  <SelectTrigger id="rag-extractor">
                    <SelectValue placeholder="Choose extractor" />
                  </SelectTrigger>
                  <SelectContent>
                    {providers?.extractors.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-ocr">OCR</Label>
                <Select
                  value={ingestionForm.ocrEngine}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, ocrEngine: value }))}
                  disabled={!ingestionForm.ocrEnabled}
                >
                  <SelectTrigger id="rag-ocr">
                    <SelectValue placeholder="Choose OCR" />
                  </SelectTrigger>
                  <SelectContent>
                    {providers?.ocr_engines.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-embedder">Embedder</Label>
                <Select
                  value={ingestionForm.embedder}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, embedder: value }))}
                >
                  <SelectTrigger id="rag-embedder">
                    <SelectValue placeholder="Choose embedder" />
                  </SelectTrigger>
                  <SelectContent>
                    {providers?.embedders.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-reranker">Reranker</Label>
                <Select
                  value={ingestionForm.reranker}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, reranker: value }))}
                  disabled={!ingestionForm.rerankerEnabled}
                >
                  <SelectTrigger id="rag-reranker">
                    <SelectValue placeholder="Choose reranker" />
                  </SelectTrigger>
                  <SelectContent>
                    {providers?.rerankers.map((provider) => (
                      <SelectItem key={provider.id} value={provider.id}>
                        {provider.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Extractor instance</Label>
                <RAGModuleSelector
                  kind="extractor"
                  catalog={moduleCatalog}
                  instances={moduleInstances}
                  value={ingestionForm.extractorInstanceId}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, extractorInstanceId: value }))}
                  disabled={!ingestionForm.extractorEnabled}
                />
              </div>
              <div className="space-y-2">
                <Label>OCR instance</Label>
                <RAGModuleSelector
                  kind="ocr"
                  catalog={moduleCatalog}
                  instances={moduleInstances}
                  value={ingestionForm.ocrInstanceId}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, ocrInstanceId: value }))}
                  disabled={!ingestionForm.ocrEnabled}
                />
              </div>
              <div className="space-y-2">
                <Label>Embedder instance</Label>
                <RAGModuleSelector
                  kind="embedder"
                  catalog={moduleCatalog}
                  instances={moduleInstances}
                  value={ingestionForm.embedderInstanceId}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, embedderInstanceId: value }))}
                />
              </div>
              <div className="space-y-2">
                <Label>Reranker instance</Label>
                <RAGModuleSelector
                  kind="reranker"
                  catalog={moduleCatalog}
                  instances={moduleInstances}
                  value={ingestionForm.rerankerInstanceId}
                  onValueChange={(value) => setIngestionForm((current) => ({ ...current, rerankerInstanceId: value }))}
                  disabled={!ingestionForm.rerankerEnabled}
                />
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-chunk-size">Chunk size</Label>
                <Input
                  id="rag-chunk-size"
                  type="number"
                  value={ingestionForm.chunkSize}
                  onChange={(event) => setIngestionForm((current) => ({ ...current, chunkSize: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-chunk-overlap">Chunk overlap</Label>
                <Input
                  id="rag-chunk-overlap"
                  type="number"
                  value={ingestionForm.chunkOverlap}
                  onChange={(event) => setIngestionForm((current) => ({ ...current, chunkOverlap: event.target.value }))}
                />
              </div>
            </div>
            <Button type="submit" disabled={busyAction === "create-ingestion-profile"}>
              {busyAction === "create-ingestion-profile" ? <Spinner className="size-4" /> : "Create Ingestion Profile"}
            </Button>
          </form>

          <Separator className="my-4" />

          <div className="space-y-3">
            {ingestionProfiles.length > 0 ? ingestionProfiles.map((item) => (
              <div key={item.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <div className="text-sm font-semibold">{item.name}</div>
                  <Badge variant="outline">{item.extractor}</Badge>
                  <Badge variant="outline">{item.ocr_engine}</Badge>
                  <Badge variant="outline">{item.embedder}</Badge>
                  {item.embedder_instance_id ? <Badge variant="secondary">instance-bound</Badge> : null}
                  {!item.ocr_enabled ? <Badge variant="secondary">ocr off</Badge> : null}
                  {item.reranker_enabled ? <Badge variant="outline">reranker on</Badge> : null}
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  Chunk recipe: {item.chunk_size} / {item.chunk_overlap}
                </div>
              </div>
            )) : (
              <Empty>
                <EmptyHeader>
                  <EmptyMedia variant="icon">
                    <HugeiconsIcon icon={Book03Icon} className="size-5" />
                  </EmptyMedia>
                  <EmptyTitle>No ingestion profiles yet</EmptyTitle>
                  <EmptyDescription>Create one profile with extraction and embedding defaults.</EmptyDescription>
                </EmptyHeader>
              </Empty>
            )}
          </div>
        </SectionCard>
      </div>
    </div>
  );

  const renderModulesTab = () => (
    <div className="flex min-w-0 flex-col gap-4 md:gap-6">
      <SectionCard
        title="Module Workspace"
        description="Search, check, install and test the concrete RAG components available on this server."
        icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
        badge={moduleCatalog.length > 0 ? `${moduleInstances.length}/${moduleCatalog.length}` : undefined}
      >
        <RAGModuleCatalogBrowser
          catalog={moduleCatalog}
          installations={moduleInstallations}
          instances={moduleInstances}
          busyAction={busyAction}
          onCheckModule={(moduleId) => void onCheckModule(moduleId)}
          onInstallModule={(moduleId) => void onInstallModule(moduleId)}
          onTestInstance={(instanceId) => void onTestModuleInstance(instanceId)}
        />
      </SectionCard>

      <div className="grid gap-4 md:gap-6 xl:grid-cols-2">
        <SectionCard
          title="Module Instances"
          description="Bind a catalog module to a concrete local path, service endpoint or system binary."
          icon={<HugeiconsIcon icon={Book03Icon} className="size-5" />}
          badge={moduleInstances.length > 0 ? String(moduleInstances.length) : undefined}
        >
          <form className="space-y-4" onSubmit={onCreateModuleInstance}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-module-instance-name">Instance name</Label>
                <Input
                  id="rag-module-instance-name"
                  value={moduleForm.name}
                  onChange={(event) => setModuleForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="Local BGE-M3"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-module-id">Module</Label>
                <Select
                  value={moduleForm.moduleId}
                  onValueChange={(value) => {
                    const nextModule = moduleCatalogById.get(value);
                    setModuleForm((current) => ({
                      ...current,
                      moduleId: value,
                      sourceType: nextModule?.source_type ?? current.sourceType,
                      modelId: current.modelId || nextModule?.default_model_id || "",
                    }));
                  }}
                >
                  <SelectTrigger id="rag-module-id">
                    <SelectValue placeholder="Choose module" />
                  </SelectTrigger>
                  <SelectContent>
                    {moduleCatalog.filter((item) => item.configurable).map((item) => (
                      <SelectItem key={item.id} value={item.id}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-module-source">Source type</Label>
              <Select
                value={moduleForm.sourceType}
                onValueChange={(value: RagModuleSourceType) =>
                  setModuleForm((current) => ({ ...current, sourceType: value }))
                }
              >
                <SelectTrigger id="rag-module-source">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(["local_model_path", "service_url", "system_binary", "python_package"] satisfies RagModuleSourceType[]).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedModule ? (
                <div className="text-xs text-muted-foreground">
                  Default source: {selectedModule.source_type}
                  {selectedModule.default_model_id ? ` · model: ${selectedModule.default_model_id}` : ""}
                </div>
              ) : null}
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-module-model">Model ID</Label>
                <Input
                  id="rag-module-model"
                  value={moduleForm.modelId}
                  onChange={(event) => setModuleForm((current) => ({ ...current, modelId: event.target.value }))}
                  placeholder="BAAI/bge-m3"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-module-path">Local path</Label>
                <Input
                  id="rag-module-path"
                  value={moduleForm.localPath}
                  onChange={(event) => setModuleForm((current) => ({ ...current, localPath: event.target.value }))}
                  placeholder="/models/bge-m3"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-module-service">Service URL</Label>
                <Input
                  id="rag-module-service"
                  value={moduleForm.serviceUrl}
                  onChange={(event) => setModuleForm((current) => ({ ...current, serviceUrl: event.target.value }))}
                  placeholder="http://127.0.0.1:6333"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-module-binary">Binary path</Label>
                <Input
                  id="rag-module-binary"
                  value={moduleForm.binaryPath}
                  onChange={(event) => setModuleForm((current) => ({ ...current, binaryPath: event.target.value }))}
                  placeholder="tesseract"
                />
              </div>
            </div>
            <Button type="submit" disabled={busyAction === "create-module-instance"}>
              {busyAction === "create-module-instance" ? <Spinner className="size-4" /> : "Create Module Instance"}
            </Button>
          </form>
        </SectionCard>

        <SectionCard
          title="Offline Package Install"
          description="Use this when the server has no network and packages are copied manually."
          icon={<HugeiconsIcon icon={DocumentAttachmentIcon} className="size-5" />}
        >
          <form className="space-y-4" onSubmit={onInstallLocalModule}>
            <div className="space-y-2">
              <Label htmlFor="rag-local-install-module">Module</Label>
              <Select
                value={localInstallForm.moduleId}
                onValueChange={(value) => setLocalInstallForm((current) => ({ ...current, moduleId: value }))}
              >
                <SelectTrigger id="rag-local-install-module">
                  <SelectValue placeholder="Choose installable module" />
                </SelectTrigger>
                <SelectContent>
                  {moduleCatalog.filter((item) => item.package_name).map((item) => (
                    <SelectItem key={item.id} value={item.id}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-local-package-path">Wheel file path</Label>
              <Input
                id="rag-local-package-path"
                value={localInstallForm.packagePath}
                onChange={(event) => setLocalInstallForm((current) => ({ ...current, packagePath: event.target.value }))}
                placeholder="/opt/packages/docling-*.whl"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-local-wheelhouse-path">Wheelhouse path</Label>
              <Input
                id="rag-local-wheelhouse-path"
                value={localInstallForm.wheelhousePath}
                onChange={(event) => setLocalInstallForm((current) => ({ ...current, wheelhousePath: event.target.value }))}
                placeholder="/opt/wheelhouse"
              />
            </div>
            <Button type="submit" disabled={busyAction === "module-install-local"}>
              {busyAction === "module-install-local" ? <Spinner className="size-4" /> : "Install From Local Path"}
            </Button>
          </form>
        </SectionCard>
      </div>
    </div>
  );

  const renderDatasetsTab = () => (
    <div className="flex min-w-0 flex-col gap-4 md:gap-6">
      <div className="grid gap-4 md:gap-6 xl:grid-cols-2">
        <SectionCard
          title="Create Dataset"
          description="Register a new `RAG dataset` before adding normalized text or files."
          icon={<HugeiconsIcon icon={DocumentAttachmentIcon} className="size-5" />}
        >
          <form className="space-y-4" onSubmit={onCreateDataset}>
            <div className="space-y-2">
              <Label htmlFor="rag-dataset-name">Dataset name</Label>
              <Input
                id="rag-dataset-name"
                value={datasetForm.name}
                onChange={(event) => setDatasetForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="April Product Docs"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-dataset-kind">Source kind</Label>
              <Select
                value={datasetForm.sourceKind}
                onValueChange={(value: "documents" | "normalized-text") =>
                  setDatasetForm((current) => ({ ...current, sourceKind: value }))
                }
              >
                <SelectTrigger id="rag-dataset-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="documents">documents</SelectItem>
                  <SelectItem value="normalized-text">normalized-text</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-dataset-description">Description</Label>
              <Textarea
                id="rag-dataset-description"
                value={datasetForm.description}
                onChange={(event) => setDatasetForm((current) => ({ ...current, description: event.target.value }))}
                placeholder="Corpus for support and operator knowledge."
              />
            </div>
            <Button type="submit" disabled={busyAction === "create-dataset"}>
              {busyAction === "create-dataset" ? <Spinner className="size-4" /> : "Create RAG Dataset"}
            </Button>
          </form>
        </SectionCard>

        <SectionCard
          title="Add Content"
          description="Choose one input mode at a time to keep the flow focused."
          icon={<HugeiconsIcon icon={Book03Icon} className="size-5" />}
        >
          <Tabs defaultValue="append-text" className="gap-4">
            <TabsList variant="line">
              <TabsTrigger value="append-text">Append Text</TabsTrigger>
              <TabsTrigger value="upload-file">Upload File</TabsTrigger>
            </TabsList>

            <TabsContent value="append-text" className="pt-2">
              <form className="space-y-4" onSubmit={onAppendText}>
                <div className="space-y-2">
                  <Label htmlFor="rag-append-dataset">Dataset</Label>
                  <Select
                    value={appendTextForm.datasetId}
                    onValueChange={(value) => setAppendTextForm((current) => ({ ...current, datasetId: value }))}
                  >
                    <SelectTrigger id="rag-append-dataset">
                      <SelectValue placeholder="Choose dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="rag-append-name">Document name</Label>
                  <Input
                    id="rag-append-name"
                    value={appendTextForm.documentName}
                    onChange={(event) => setAppendTextForm((current) => ({ ...current, documentName: event.target.value }))}
                    placeholder="release-notes.md"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="rag-append-text">Text</Label>
                  <Textarea
                    id="rag-append-text"
                    rows={7}
                    value={appendTextForm.text}
                    onChange={(event) => setAppendTextForm((current) => ({ ...current, text: event.target.value }))}
                    placeholder="Paste normalized content here."
                  />
                </div>
                <Button type="submit" disabled={busyAction === "append-text" || datasets.length === 0}>
                  {busyAction === "append-text" ? <Spinner className="size-4" /> : "Append Text"}
                </Button>
              </form>
            </TabsContent>

            <TabsContent value="upload-file" className="pt-2">
              <form className="space-y-4" onSubmit={onUploadDocument}>
                <div className="space-y-2">
                  <Label htmlFor="rag-upload-dataset">Dataset</Label>
                  <Select value={uploadDatasetId} onValueChange={setUploadDatasetId}>
                    <SelectTrigger id="rag-upload-dataset">
                      <SelectValue placeholder="Choose dataset" />
                    </SelectTrigger>
                    <SelectContent>
                      {datasets.map((dataset) => (
                        <SelectItem key={dataset.id} value={dataset.id}>
                          {dataset.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="rag-upload-file">File</Label>
                  <Input id="rag-upload-file" type="file" onChange={onFileChange} />
                </div>
                <Button type="submit" disabled={busyAction === "upload-document" || !uploadDatasetId}>
                  {busyAction === "upload-document" ? <Spinner className="size-4" /> : "Upload Document"}
                </Button>
              </form>
            </TabsContent>
          </Tabs>
        </SectionCard>
      </div>

      <SectionCard
        title="Dataset Library"
        description="Current corpora with aggregate counts, readiness and input type."
        icon={<HugeiconsIcon icon={DocumentAttachmentIcon} className="size-5" />}
        badge={datasets.length > 0 ? String(datasets.length) : undefined}
      >
        <div className="space-y-3">
          {datasets.length > 0 ? datasets.map((dataset) => (
            <div key={dataset.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <div className="text-sm font-semibold">{dataset.name}</div>
                <Badge variant="outline">{dataset.source_kind}</Badge>
                <Badge variant="outline">{dataset.status}</Badge>
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                {metricLabel(dataset.documents_count)} documents · {metricLabel(dataset.chunks_count)} chunks · {metricLabel(dataset.total_characters)} chars
              </div>
              {dataset.description ? <div className="mt-2 text-xs text-muted-foreground">{dataset.description}</div> : null}
            </div>
          )) : (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <HugeiconsIcon icon={DocumentAttachmentIcon} className="size-5" />
                </EmptyMedia>
                <EmptyTitle>No datasets yet</EmptyTitle>
                <EmptyDescription>Create a corpus first, then add normalized text or upload files.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
      </SectionCard>
    </div>
  );

  const renderCollectionsTab = () => (
    <div className="flex min-w-0 flex-col gap-4 md:gap-6">
      <div className="grid gap-4 md:gap-6 xl:grid-cols-2">
        <SectionCard
          title="Create Collection"
          description="Bind a logical knowledge target to a connection profile and ingestion profile."
          icon={<HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-5" />}
        >
          <form className="space-y-4" onSubmit={onCreateCollection}>
            <div className="space-y-2">
              <Label htmlFor="rag-collection-name">Collection name</Label>
              <Input
                id="rag-collection-name"
                value={collectionForm.name}
                onChange={(event) => setCollectionForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Support Knowledge"
              />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-collection-connection">Connection profile</Label>
                <Select
                  value={collectionForm.connectionProfileId}
                  onValueChange={(value) => setCollectionForm((current) => ({ ...current, connectionProfileId: value }))}
                >
                  <SelectTrigger id="rag-collection-connection">
                    <SelectValue placeholder="Choose connection" />
                  </SelectTrigger>
                  <SelectContent>
                    {connectionProfiles.map((profile) => (
                      <SelectItem key={profile.id} value={profile.id}>
                        {profile.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-collection-ingestion">Ingestion profile</Label>
                <Select
                  value={collectionForm.ingestionProfileId}
                  onValueChange={(value) => setCollectionForm((current) => ({ ...current, ingestionProfileId: value }))}
                >
                  <SelectTrigger id="rag-collection-ingestion">
                    <SelectValue placeholder="Choose ingestion profile" />
                  </SelectTrigger>
                  <SelectContent>
                    {ingestionProfiles.map((profile) => (
                      <SelectItem key={profile.id} value={profile.id}>
                        {profile.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="rag-remote-collection">Remote collection name</Label>
              <Input
                id="rag-remote-collection"
                value={collectionForm.remoteCollectionName}
                onChange={(event) => setCollectionForm((current) => ({ ...current, remoteCollectionName: event.target.value }))}
                placeholder="support_knowledge"
              />
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-semibold">Ingestion module preview</div>
                  <div className="text-xs text-muted-foreground">
                    The collection will use this profile composition during publish and reindex.
                  </div>
                </div>
                {selectedCollectionIngestionProfile ? (
                  <Badge variant="outline">{selectedCollectionIngestionProfile.name}</Badge>
                ) : (
                  <Badge variant="secondary">No profile</Badge>
                )}
              </div>
              {selectedProfileModules.length > 0 ? (
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {selectedProfileModules.map((item) => (
                    <div key={`${item.kind}:${item.label}`} className="rounded-xl border border-border/60 bg-background/70 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="min-w-0">
                          <div className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">{item.label}</div>
                          <div className="truncate text-sm font-medium">{item.name}</div>
                        </div>
                        <Badge variant={moduleStatusVariant(item.status)}>{item.status}</Badge>
                      </div>
                      <div className="mt-1 truncate text-xs text-muted-foreground">{item.target}</div>
                      {item.warning ? (
                        <div className="mt-2 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1 text-[11px] text-destructive">
                          {item.warning}
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="mt-3 text-xs text-muted-foreground">
                  Choose an ingestion profile to preview concrete modules.
                </div>
              )}
            </div>
            <Button type="submit" disabled={busyAction === "create-collection"}>
              {busyAction === "create-collection" ? <Spinner className="size-4" /> : "Create Collection"}
            </Button>
          </form>
        </SectionCard>

        <SectionCard
          title="Publish Dataset"
          description="Send a prepared corpus into the selected collection and create or update the active projection."
          icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
        >
          <form className="space-y-4" onSubmit={onPublishDataset}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rag-publish-collection">Collection</Label>
                <Select
                  value={publishForm.collectionId}
                  onValueChange={(value) => setPublishForm((current) => ({ ...current, collectionId: value }))}
                >
                  <SelectTrigger id="rag-publish-collection">
                    <SelectValue placeholder="Choose collection" />
                  </SelectTrigger>
                  <SelectContent>
                    {collections.map((collection) => (
                      <SelectItem key={collection.id} value={collection.id}>
                        {collection.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="rag-publish-dataset">Dataset</Label>
                <Select
                  value={publishForm.datasetId}
                  onValueChange={(value) => setPublishForm((current) => ({ ...current, datasetId: value }))}
                >
                  <SelectTrigger id="rag-publish-dataset">
                    <SelectValue placeholder="Choose dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((dataset) => (
                      <SelectItem key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
            <Button type="submit" disabled={busyAction === "publish-dataset" || collections.length === 0 || datasets.length === 0}>
              {busyAction === "publish-dataset" ? <Spinner className="size-4" /> : "Publish Dataset"}
            </Button>
          </form>
        </SectionCard>
      </div>

      <SectionCard
        title="Collections"
        description="Active targets with projection details, counts and per-collection operational actions."
        icon={<HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-5" />}
        badge={collections.length > 0 ? String(collections.length) : undefined}
      >
        <div className="space-y-3">
          {collections.length > 0 ? collections.map((item) => {
            const reindexKey = `reindex:${item.id}`;
            const syncKey = `sync:${item.id}`;
            return (
              <div key={item.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-sm font-semibold">{item.name}</div>
                      <Badge variant="outline">{item.backend}</Badge>
                      <Badge variant={syncStatusVariant(item.sync_status)}>{item.sync_status}</Badge>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {item.connection_profile_name} · {item.ingestion_profile_name} · {item.remote_collection_name}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={busyAction === reindexKey}
                      onClick={() => void onReindex(item.id)}
                    >
                      {busyAction === reindexKey ? <Spinner className="size-4" /> : "Reindex"}
                    </Button>
                    <Button
                      size="sm"
                      disabled={busyAction === syncKey}
                      onClick={() => void onSync(item.id)}
                    >
                      {busyAction === syncKey ? <Spinner className="size-4" /> : "Sync Open WebUI"}
                    </Button>
                  </div>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Documents</div>
                    <div className="mt-1 text-sm font-medium">{metricLabel(item.documents_count)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Chunks</div>
                    <div className="mt-1 text-sm font-medium">{metricLabel(item.chunks_count)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Projection</div>
                    <div className="mt-1 text-sm font-medium">v{item.active_projection.version}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Chunk recipe</div>
                    <div className="mt-1 text-sm font-medium">{item.active_projection.chunk_recipe}</div>
                  </div>
                </div>
              </div>
            );
          }) : (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <HugeiconsIcon icon={CheckmarkCircle02Icon} className="size-5" />
                </EmptyMedia>
                <EmptyTitle>No collections yet</EmptyTitle>
                <EmptyDescription>Create a collection after you have a connection profile and an ingestion profile.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
      </SectionCard>

      <SectionCard
        title="Connection Diagnostics"
        description="Secondary operational status for external targets. Keep this below publish flows so it does not dominate the page."
        icon={<HugeiconsIcon icon={Search01Icon} className="size-5" />}
      >
        <div className="space-y-3">
          {diagnostics.length > 0 ? diagnostics.map((item) => (
            <div key={item.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-semibold">{item.name}</div>
                  <div className="text-xs text-muted-foreground">{item.backend}</div>
                </div>
                <Badge variant={connectionStatusVariant(item.status)}>{item.status}</Badge>
              </div>
            </div>
          )) : (
            <div className="text-sm text-muted-foreground">No diagnostics available yet.</div>
          )}
        </div>
      </SectionCard>
    </div>
  );

  const renderJobsTab = () => (
    <SectionCard
      title="Jobs"
      description="Latest publish, reindex and Open WebUI sync operations."
      icon={<HugeiconsIcon icon={Search01Icon} className="size-5" />}
      badge={jobs.length > 0 ? String(jobs.length) : undefined}
    >
      <div className="grid gap-3 sm:grid-cols-3">
        <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
          <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Running</div>
          <div className="mt-1 text-xl font-semibold tracking-tight">{metricLabel(runningJobs)}</div>
        </div>
        <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
          <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Queued</div>
          <div className="mt-1 text-xl font-semibold tracking-tight">{metricLabel(queuedJobs)}</div>
        </div>
        <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
          <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">Errors</div>
          <div className="mt-1 text-xl font-semibold tracking-tight">{metricLabel(erroredJobs)}</div>
        </div>
      </div>

      <Separator className="my-4" />

      <div className="space-y-3">
        {jobs.length > 0 ? jobs.map((item) => (
          <div key={item.id} className="rounded-2xl border border-border/70 bg-background/80 p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <div className="text-sm font-semibold">{item.collection_name}</div>
                  <Badge variant={jobStatusVariant(item.status)}>{item.status}</Badge>
                  <Badge variant="outline">{item.job_type}</Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {item.stage_label}
                  {item.dataset_name ? ` · ${item.dataset_name}` : ""}
                </div>
              </div>
              <div className="text-xs text-muted-foreground">{item.created_at}</div>
            </div>
          </div>
        )) : (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <HugeiconsIcon icon={Search01Icon} className="size-5" />
              </EmptyMedia>
              <EmptyTitle>No jobs yet</EmptyTitle>
              <EmptyDescription>Jobs will appear here after the first publish or reindex run.</EmptyDescription>
            </EmptyHeader>
            <EmptyContent />
          </Empty>
        )}
      </div>
    </SectionCard>
  );

  return (
    <div className="relative min-h-screen bg-background">
      <main className="relative z-10 mx-auto max-w-7xl px-4 py-4 sm:px-6">
        <div className="mb-6 flex flex-col gap-0.5 sm:mb-8">
          <h1 className="text-2xl font-semibold tracking-tight">RAG Admin</h1>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
        </div>

        <SectionCard
          title="RAG Control Plane"
          description="Profiles, corpora, collections, publish jobs and Open WebUI compatibility."
          icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
          featured
          accent="blue"
        >
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            {metrics.map((item) => (
              <div
                key={item.label}
                className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3"
              >
                <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                  {item.label}
                </div>
                <div className="mt-1 text-xl font-semibold tracking-tight">
                  {metricLabel(item.value)}
                </div>
              </div>
            ))}
          </div>

          <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,0.9fr)_minmax(0,1.2fr)]">
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                Connection Health
              </div>
              <div className="mt-1 text-xl font-semibold tracking-tight">
                {metricLabel(healthyConnections)}
                <span className="text-sm font-medium text-muted-foreground"> / {metricLabel(connectionProfiles.length)}</span>
              </div>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                Open WebUI Sync
              </div>
              <div className="mt-1 text-xl font-semibold tracking-tight">
                {metricLabel(overview?.synced_collections_total ?? 0)}
              </div>
              <div className="text-xs text-muted-foreground">collections currently marked as synced</div>
            </div>
            <div className="rounded-2xl border border-border/70 bg-background/80 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.08em] text-muted-foreground">
                Supported Backends
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                {supportedBackends}
              </div>
            </div>
          </div>
        </SectionCard>

        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as AdminRagTab)}
          className="mt-6"
        >
          <TabsList variant="line">
            <TabsTrigger value="configure">Configure</TabsTrigger>
            <TabsTrigger value="modules">Modules</TabsTrigger>
            <TabsTrigger value="datasets">Datasets</TabsTrigger>
            <TabsTrigger value="collections">Collections</TabsTrigger>
            <TabsTrigger value="jobs">Jobs</TabsTrigger>
          </TabsList>

          <TabsContent value="configure" className="pt-4 md:pt-6">
            {renderConfigureTab()}
          </TabsContent>

          <TabsContent value="modules" className="pt-4 md:pt-6">
            {renderModulesTab()}
          </TabsContent>

          <TabsContent value="datasets" className="pt-4 md:pt-6">
            {renderDatasetsTab()}
          </TabsContent>

          <TabsContent value="collections" className="pt-4 md:pt-6">
            {renderCollectionsTab()}
          </TabsContent>

          <TabsContent value="jobs" className="pt-4 md:pt-6">
            {renderJobsTab()}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
