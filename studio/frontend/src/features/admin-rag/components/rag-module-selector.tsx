// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState, type ReactNode } from "react";
import type {
  RagModuleCatalogItem,
  RagModuleInstallationSummary,
  RagModuleInstanceSummary,
  RagModuleKind,
  RagModuleStatus,
} from "../api/rag-api";

type CatalogAction = (moduleId: string) => void;
type InstanceAction = (instanceId: string) => void;

const STATUS_LABELS: Record<RagModuleStatus, string> = {
  available: "available",
  missing: "missing",
  configured: "configured",
  error: "error",
  disabled: "disabled",
};

export function moduleStatusVariant(
  status: RagModuleStatus,
): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "available":
    case "configured":
      return "default";
    case "missing":
    case "disabled":
      return "secondary";
    case "error":
      return "destructive";
    default:
      return "outline";
  }
}

function moduleStatusDot(status: RagModuleStatus): string {
  switch (status) {
    case "available":
    case "configured":
      return "bg-emerald-500";
    case "missing":
    case "disabled":
      return "bg-muted-foreground/40";
    case "error":
      return "bg-destructive";
    default:
      return "bg-muted-foreground/40";
  }
}

function normalizeForSearch(value: string): string {
  return value.toLowerCase().replace(/[\s\-_./:]/g, "");
}

function ListLabel({
  children,
  count,
}: {
  children: ReactNode;
  count: number;
}) {
  return (
    <div className="flex items-center justify-between px-2.5 py-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {children}
      </span>
      <span className="text-[10px] text-muted-foreground">{count}</span>
    </div>
  );
}

function moduleTarget(instance: RagModuleInstanceSummary): string {
  return (
    instance.local_path
    || instance.model_id
    || instance.service_url
    || instance.binary_path
    || instance.module_id
  );
}

function catalogTarget(item: RagModuleCatalogItem, installation?: RagModuleInstallationSummary): string {
  return (
    installation?.path
    || item.default_model_id
    || item.package_name
    || item.binary_name
    || item.source_type
  );
}

function searchableText(values: Array<string | null | undefined>): string {
  return normalizeForSearch(values.filter(Boolean).join(" "));
}

function EmptyRows({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-[8px] px-2.5 py-3 text-xs text-muted-foreground">
      {children}
    </div>
  );
}

function ModuleRow({
  title,
  meta,
  status,
  selected,
  onClick,
  children,
}: {
  title: string;
  meta: string;
  status: RagModuleStatus;
  selected?: boolean;
  onClick?: () => void;
  children?: ReactNode;
}) {
  const content = (
    <>
      <span className={cn("size-2 shrink-0 rounded-full", moduleStatusDot(status))} />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">{title}</span>
        <span className="block truncate text-[11px] text-muted-foreground">{meta}</span>
      </span>
      <span className="ml-auto flex shrink-0 items-center gap-1.5">
        <Badge variant={moduleStatusVariant(status)}>{STATUS_LABELS[status]}</Badge>
        {children}
      </span>
    </>
  );

  if (!onClick) {
    return (
      <div className="flex w-full items-center gap-2 rounded-[6px] px-2.5 py-2 text-left text-sm">
        {content}
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-2 rounded-[6px] px-2.5 py-2 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[#2e3035]",
        selected && "bg-[#ececec] dark:bg-[#2e3035]",
      )}
    >
      {content}
    </button>
  );
}

export function RAGModuleSelector({
  kind,
  catalog,
  instances,
  value,
  onValueChange,
  placeholder = "Use profile default",
  disabled = false,
}: {
  kind: RagModuleKind;
  catalog: RagModuleCatalogItem[];
  instances: RagModuleInstanceSummary[];
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const selected = instances.find((item) => item.id === value);
  const kindCatalog = catalog.filter((item) => item.kind === kind);
  const normalizedQuery = normalizeForSearch(query);
  const filteredInstances = instances.filter((item) => {
    if (item.kind !== kind) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }
    return searchableText([
      item.name,
      item.module_id,
      item.source_type,
      moduleTarget(item),
      item.status,
    ]).includes(normalizedQuery);
  });

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex h-9 w-full items-center gap-2 rounded-[8px] border border-border/60 px-3 text-left text-sm transition-colors hover:bg-[#ececec] disabled:cursor-not-allowed disabled:opacity-50 dark:hover:bg-[#2e3035]",
            selected && "bg-background/80",
          )}
        >
          {selected ? (
            <span className={cn("size-2 shrink-0 rounded-full", moduleStatusDot(selected.status))} />
          ) : null}
          <span className="min-w-0 flex-1 truncate font-medium">
            {selected?.name ?? placeholder}
          </span>
          {selected ? (
            <span className="hidden max-w-[45%] truncate text-xs text-muted-foreground sm:block">
              {moduleTarget(selected)}
            </span>
          ) : (
            <span className="hidden text-xs text-muted-foreground sm:block">
              {kindCatalog.length} available
            </span>
          )}
          <HugeiconsIcon icon={ArrowDown01Icon} className="size-3.5 shrink-0 text-muted-foreground" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[min(440px,calc(100vw-1rem))] gap-2 p-2">
        <div className="relative">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="h-8 pl-8"
            placeholder="Search configured modules"
          />
        </div>

        <div className="max-h-[320px] overflow-y-auto pr-1">
          <ListLabel count={filteredInstances.length}>Configured {kind}</ListLabel>
          <ModuleRow
            title="Use profile default"
            meta="Fallback to provider selected in the profile"
            status="available"
            selected={!value}
            onClick={() => {
              onValueChange("");
              setOpen(false);
            }}
          />
          {filteredInstances.length > 0 ? (
            filteredInstances.map((instance) => (
              <ModuleRow
                key={instance.id}
                title={instance.name}
                meta={`${instance.module_id} · ${moduleTarget(instance)}`}
                status={instance.status}
                selected={instance.id === value}
                onClick={() => {
                  onValueChange(instance.id);
                  setOpen(false);
                }}
              />
            ))
          ) : (
            <EmptyRows>No configured instances for this module type.</EmptyRows>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function RAGModuleCatalogBrowser({
  catalog,
  installations,
  instances,
  busyAction,
  onCheckModule,
  onInstallModule,
  onTestInstance,
}: {
  catalog: RagModuleCatalogItem[];
  installations: RagModuleInstallationSummary[];
  instances: RagModuleInstanceSummary[];
  busyAction: string | null;
  onCheckModule: CatalogAction;
  onInstallModule: CatalogAction;
  onTestInstance: InstanceAction;
}) {
  const [query, setQuery] = useState("");
  const installationById = useMemo(
    () => new Map(installations.map((item) => [item.module_id, item])),
    [installations],
  );
  const normalizedQuery = normalizeForSearch(query);

  const filteredInstances = instances.filter((instance) => {
    if (!normalizedQuery) {
      return true;
    }
    return searchableText([
      instance.name,
      instance.module_id,
      instance.kind,
      instance.source_type,
      moduleTarget(instance),
      instance.status,
    ]).includes(normalizedQuery);
  });

  const filteredCatalog = catalog.filter((item) => {
    if (!normalizedQuery) {
      return true;
    }
    const installation = installationById.get(item.id);
    return searchableText([
      item.label,
      item.id,
      item.kind,
      item.source_type,
      item.package_name,
      item.module_name,
      item.default_model_id,
      installation?.status,
      installation?.path,
      installation?.last_error,
    ]).includes(normalizedQuery);
  });

  const available = filteredCatalog.filter((item) => {
    const status = installationById.get(item.id)?.status ?? (item.enabled ? "available" : "disabled");
    return status === "available" || status === "configured";
  });
  const missing = filteredCatalog.filter((item) => {
    const status = installationById.get(item.id)?.status ?? (item.enabled ? "available" : "disabled");
    return status === "missing" || status === "error" || status === "disabled";
  });

  const renderCatalogRow = (item: RagModuleCatalogItem) => {
    const installation = installationById.get(item.id);
    const status = installation?.status ?? (item.enabled ? "available" : "disabled");
    const checkKey = `module-check:${item.id}`;
    const installKey = `module-install:${item.id}`;
    return (
      <div key={item.id} className="rounded-[8px] border border-border/60 bg-background/70 p-2">
        <ModuleRow
          title={item.label}
          meta={`${item.kind} · ${catalogTarget(item, installation)}`}
          status={status}
        >
          <span className="hidden text-[10px] text-muted-foreground sm:inline">{item.source_type}</span>
        </ModuleRow>
        {installation?.last_error ? (
          <div className="mx-2 mb-2 rounded-md border border-destructive/30 bg-destructive/10 px-2 py-1.5 text-[11px] text-destructive">
            {installation.last_error}
          </div>
        ) : null}
        <div className="flex flex-wrap gap-2 px-2 pb-1">
          <Button
            variant="outline"
            size="sm"
            disabled={busyAction === checkKey}
            onClick={() => onCheckModule(item.id)}
          >
            {busyAction === checkKey ? <Spinner className="size-4" /> : "Check"}
          </Button>
          <Button
            size="sm"
            disabled={!item.installable || busyAction === installKey}
            onClick={() => onInstallModule(item.id)}
          >
            {busyAction === installKey ? <Spinner className="size-4" /> : "Install"}
          </Button>
          {item.dependencies.map((dependency) => (
            <Badge key={dependency} variant="secondary">{dependency}</Badge>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-3">
      <div className="relative max-w-xl">
        <HugeiconsIcon
          icon={Search01Icon}
          className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
        />
        <Input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          className="pl-8"
          placeholder="Search modules, packages, paths and statuses"
        />
      </div>

      <div className="grid gap-3 xl:grid-cols-3">
        <div className="space-y-2 rounded-2xl border border-border/70 bg-background/80 p-3">
          <ListLabel count={filteredInstances.length}>Installed / Configured</ListLabel>
          {filteredInstances.length > 0 ? filteredInstances.map((instance) => {
            const testKey = `module-instance-test:${instance.id}`;
            return (
              <div key={instance.id} className="rounded-[8px] border border-border/60 bg-background/70 p-2">
                <ModuleRow
                  title={instance.name}
                  meta={`${instance.kind} · ${moduleTarget(instance)}`}
                  status={instance.status}
                />
                {instance.last_error ? (
                  <div className="mx-2 mb-2 text-[11px] text-destructive">{instance.last_error}</div>
                ) : null}
                <div className="px-2 pb-1">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={busyAction === testKey}
                    onClick={() => onTestInstance(instance.id)}
                  >
                    {busyAction === testKey ? <Spinner className="size-4" /> : "Test Instance"}
                  </Button>
                </div>
              </div>
            );
          }) : (
            <EmptyRows>No configured module instances.</EmptyRows>
          )}
        </div>

        <div className="space-y-2 rounded-2xl border border-border/70 bg-background/80 p-3">
          <ListLabel count={available.length}>Available</ListLabel>
          {available.length > 0 ? available.map(renderCatalogRow) : (
            <EmptyRows>No available modules match the search.</EmptyRows>
          )}
        </div>

        <div className="space-y-2 rounded-2xl border border-border/70 bg-background/80 p-3">
          <ListLabel count={missing.length}>Missing / Disabled</ListLabel>
          {missing.length > 0 ? missing.map(renderCatalogRow) : (
            <EmptyRows>No missing modules match the search.</EmptyRows>
          )}
        </div>
      </div>
    </div>
  );
}
