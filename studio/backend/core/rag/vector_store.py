# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any, Protocol

import httpx


class VectorStoreAdapterProtocol(Protocol):
    def health_check(self) -> dict[str, object]: ...

    def get_collection_info(self, *, collection_name: str) -> dict[str, object]: ...

    def scroll_points(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: object | None = None,
    ) -> dict[str, object]: ...

    def search_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
    ) -> dict[str, object]: ...

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

    def get_collection_info(self, *, collection_name: str) -> dict[str, object]:
        response = httpx.get(
            f"{self._base_url}/collections/{collection_name}",
            headers = self._headers,
            timeout = self._timeout,
        )
        if response.status_code == 404:
            return {
                "status": "missing",
                "details": {"collection_name": collection_name, "base_url": self._base_url},
            }
        response.raise_for_status()
        payload = response.json()
        return {
            "status": "found",
            "details": payload.get("result", {}),
        }

    def scroll_points(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: object | None = None,
    ) -> dict[str, object]:
        if isinstance(offset, int) and offset > 0:
            return self._scroll_points_by_index(
                collection_name = collection_name,
                limit = limit,
                offset = offset,
            )
        return self._scroll_points_raw(
            collection_name = collection_name,
            limit = limit,
            offset = offset,
        )

    def search_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
    ) -> dict[str, object]:
        response = httpx.post(
            f"{self._base_url}/collections/{collection_name}/points/search",
            headers = self._headers,
            json = {
                "vector": vector,
                "limit": max(1, min(limit, 100)),
                "with_payload": True,
                "with_vector": False,
            },
            timeout = self._timeout,
        )
        if response.status_code == 404:
            return {"points": [], "status": "missing"}
        response.raise_for_status()
        payload = response.json()
        return {
            "points": payload.get("result", []),
            "status": "ok",
        }

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

    def _scroll_points_raw(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: object | None,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "limit": max(1, min(limit, 1000)),
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None and offset != 0:
            body["offset"] = offset
        response = httpx.post(
            f"{self._base_url}/collections/{collection_name}/points/scroll",
            headers = self._headers,
            json = body,
            timeout = self._timeout,
        )
        if response.status_code == 404:
            return {"points": [], "next_page_offset": None, "status": "missing"}
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result", {})
        return {
            "points": result.get("points", []),
            "next_page_offset": result.get("next_page_offset"),
            "status": "ok",
        }

    def _scroll_points_by_index(
        self,
        *,
        collection_name: str,
        limit: int,
        offset: int,
    ) -> dict[str, object]:
        target_count = offset + max(1, min(limit, 1000))
        points: list[object] = []
        cursor: object | None = None
        while len(points) < target_count:
            response = self._scroll_points_raw(
                collection_name = collection_name,
                limit = min(256, target_count - len(points)),
                offset = cursor,
            )
            batch = list(response.get("points", []))
            points.extend(batch)
            cursor = response.get("next_page_offset")
            if not batch or cursor is None:
                break
        page = points[offset:target_count]
        next_offset = offset + len(page) if len(points) > target_count or cursor is not None else None
        return {
            "points": page,
            "next_page_offset": next_offset,
            "status": "ok",
        }


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
