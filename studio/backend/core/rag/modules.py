# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from models.rag import (
    RAGModuleCatalogItem,
    RAGModuleInstallationSummary,
    RAGModuleInstanceSummary,
)


def module_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond = 0).isoformat()


def check_catalog_module(item: RAGModuleCatalogItem) -> RAGModuleInstallationSummary:
    now = module_now_iso()
    if not item.enabled:
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "disabled",
            last_checked_at = now,
            last_error = None,
        )
    if item.source_type in {"builtin", "service_url", "local_model_path"} and not item.module_name:
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "available",
            path = item.default_model_id,
            last_checked_at = now,
        )
    if item.source_type == "system_binary":
        binary_path = shutil.which(item.binary_name or item.id)
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "available" if binary_path else "missing",
            path = binary_path,
            last_checked_at = now,
            last_error = None if binary_path else f"Binary `{item.binary_name or item.id}` not found in PATH.",
        )
    if item.module_name:
        spec = importlib.util.find_spec(item.module_name)
        version = _distribution_version(item.package_name or item.module_name)
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "available" if spec is not None else "missing",
            version = version,
            path = item.default_model_id,
            last_checked_at = now,
            last_error = None if spec is not None else f"Python module `{item.module_name}` is not installed.",
            install_command = _install_command(item),
        )
    return RAGModuleInstallationSummary(
        module_id = item.id,
        kind = item.kind,
        source_type = item.source_type,
        status = "missing",
        last_checked_at = now,
        last_error = "No check rule is configured for this module.",
    )


def check_module_instance(
    item: RAGModuleCatalogItem,
    instance: dict[str, Any],
) -> RAGModuleInstanceSummary:
    now = module_now_iso()
    status = "configured"
    last_error: str | None = None
    source_type = str(instance["source_type"])

    if not bool(instance.get("enabled", True)):
        status = "disabled"
    elif source_type == "local_model_path":
        local_path = str(instance.get("local_path") or "").strip()
        model_id = str(instance.get("model_id") or "").strip()
        if local_path and not Path(local_path).expanduser().exists():
            status = "error"
            last_error = f"Local path `{local_path}` does not exist."
        elif not local_path and not model_id and not item.default_model_id:
            status = "error"
            last_error = "Set either local_path or model_id."
    elif source_type == "system_binary":
        binary_path = str(instance.get("binary_path") or item.binary_name or item.id).strip()
        resolved = Path(binary_path).expanduser().exists() if "/" in binary_path else shutil.which(binary_path)
        if not resolved:
            status = "error"
            last_error = f"Binary `{binary_path}` not found."
    elif source_type == "service_url":
        service_url = str(instance.get("service_url") or "").strip()
        if not service_url:
            status = "error"
            last_error = "Service URL is required."
    elif source_type == "python_package":
        if item.module_name and importlib.util.find_spec(item.module_name) is None:
            status = "error"
            last_error = f"Python module `{item.module_name}` is not installed."

    payload = dict(instance)
    payload["kind"] = item.kind
    payload["status"] = status
    payload["last_checked_at"] = now
    payload["last_error"] = last_error
    return RAGModuleInstanceSummary.model_validate(payload)


def install_catalog_module(item: RAGModuleCatalogItem) -> RAGModuleInstallationSummary:
    if not item.installable or not item.package_name:
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "error",
            last_checked_at = module_now_iso(),
            last_error = "This module is not installable through the managed installer.",
        )
    command = [sys.executable, "-m", "pip", "install", item.package_name]
    result = _run_pip(command)
    if result["returncode"] != 0:
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "error",
            last_checked_at = module_now_iso(),
            last_error = str(result["stderr"] or result["stdout"] or "pip install failed."),
            install_command = _install_command(item),
        )
    return check_catalog_module(item)


def install_local_catalog_module(
    item: RAGModuleCatalogItem,
    *,
    package_path: str | None,
    wheelhouse_path: str | None,
) -> RAGModuleInstallationSummary:
    if not item.package_name:
        return RAGModuleInstallationSummary(
            module_id = item.id,
            kind = item.kind,
            source_type = item.source_type,
            status = "error",
            last_checked_at = module_now_iso(),
            last_error = "Local install requires a Python package-backed module.",
        )

    if package_path:
        candidate = Path(package_path).expanduser()
        if not candidate.exists():
            return _install_error(item, f"Package path `{candidate}` does not exist.")
        command = [sys.executable, "-m", "pip", "install", str(candidate)]
    elif wheelhouse_path:
        candidate = Path(wheelhouse_path).expanduser()
        if not candidate.exists():
            return _install_error(item, f"Wheelhouse path `{candidate}` does not exist.")
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(candidate),
            item.package_name,
        ]
    else:
        return _install_error(item, "Set package_path or wheelhouse_path.")

    result = _run_pip(command)
    if result["returncode"] != 0:
        return _install_error(item, str(result["stderr"] or result["stdout"] or "pip install failed."))
    return check_catalog_module(item)


def _distribution_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_command(item: RAGModuleCatalogItem) -> str | None:
    if not item.installable or not item.package_name:
        return None
    return f"{sys.executable} -m pip install {item.package_name}"


def _install_error(item: RAGModuleCatalogItem, detail: str) -> RAGModuleInstallationSummary:
    return RAGModuleInstallationSummary(
        module_id = item.id,
        kind = item.kind,
        source_type = item.source_type,
        status = "error",
        last_checked_at = module_now_iso(),
        last_error = detail,
        install_command = _install_command(item),
    )


def _run_pip(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check = False,
            capture_output = True,
            text = True,
            timeout = 600,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime process controls
        return {"returncode": 1, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }
