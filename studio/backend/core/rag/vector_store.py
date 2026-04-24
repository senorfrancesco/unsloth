# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Protocol

import httpx


class VectorStoreAdapterProtocol(Protocol):
    def health_check(self) -> dict[str, object]: ...

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None: ...

    def upsert_chunks(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]: ...

    def replace_projection(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]: ...


class QdrantVectorStoreAdapter:
    def __init__(self, *, base_url: str, api_key: str | None = None, timeout: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"api-key": api_key} if api_key else {}
        self._timeout = timeout

    def health_check(self) -> dict[str, object]:
        try:
            response = httpx.get(
                f"{self._base_url}/collections",
                headers = self._headers,
                timeout = self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
            collections = payload.get("result", {}).get("collections", [])
            return {
                "status": "healthy",
                "details": {"collections": len(collections), "base_url": self._base_url},
            }
        except Exception as exc:
            return {
                "status": "error",
                "details": {"message": str(exc), "base_url": self._base_url},
            }

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        response = httpx.get(
            f"{self._base_url}/collections/{collection_name}",
            headers = self._headers,
            timeout = self._timeout,
        )
        if response.status_code == 200:
            return
        payload = {
            "vectors": {"size": vector_size, "distance": "Cosine"},
        }
        create_response = httpx.put(
            f"{self._base_url}/collections/{collection_name}",
            headers = self._headers,
            json = payload,
            timeout = self._timeout,
        )
        create_response.raise_for_status()

    def upsert_chunks(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]:
        response = httpx.put(
            f"{self._base_url}/collections/{collection_name}/points",
            headers = self._headers,
            json = {"points": points},
            timeout = self._timeout,
        )
        response.raise_for_status()
        return {"upserted_points": len(points)}

    def replace_projection(
        self,
        *,
        collection_name: str,
        points: list[dict[str, object]],
    ) -> dict[str, object]:
        self._delete_collection_points(collection_name)
        return self.upsert_chunks(collection_name = collection_name, points = points)

    def _delete_collection_points(self, collection_name: str) -> None:
        response = httpx.post(
            f"{self._base_url}/collections/{collection_name}/points/delete",
            headers = self._headers,
            json = {"filter": {}},
            timeout = self._timeout,
        )
        response.raise_for_status()


def build_vector_store_adapter(connection_profile: dict[str, Any]) -> VectorStoreAdapterProtocol:
    backend = str(connection_profile["backend"])
    if backend != "qdrant":
        raise RuntimeError(
            f"Vector store backend `{backend}` is not implemented in this stage."
        )
    return QdrantVectorStoreAdapter(
        base_url = str(connection_profile["base_url"]),
        api_key = connection_profile.get("api_key"),
    )
