# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from models.rag import (
    RAGModuleCatalogItem,
    RAGModuleCatalogResponse,
    RAGProviderOption,
    RAGProvidersResponse,
)


def get_rag_module_catalog() -> RAGModuleCatalogResponse:
    items = [
        RAGModuleCatalogItem(
            id = "qdrant",
            kind = "vector_store",
            label = "Qdrant",
            description = "Основной self-hosted backend первого этапа.",
            enabled = True,
            stage = "first-class",
            source_type = "service_url",
            configurable = True,
            dependencies = ["qdrant-client"],
            config_schema = {"base_url": "http://127.0.0.1:6333"},
        ),
        RAGModuleCatalogItem(
            id = "pgvector",
            kind = "vector_store",
            label = "pgvector",
            description = "Закладывается в интерфейсы, но ещё не реализован.",
            enabled = False,
            stage = "design-only",
            source_type = "service_url",
            configurable = True,
            dependencies = ["psycopg"],
        ),
        RAGModuleCatalogItem(
            id = "builtin-text",
            kind = "extractor",
            label = "Builtin Text",
            description = "Встроенный путь для уже нормализованного текста и простых text/plain файлов.",
            enabled = True,
            stage = "first-class",
            source_type = "builtin",
        ),
        RAGModuleCatalogItem(
            id = "docling",
            kind = "extractor",
            label = "Docling",
            description = "Основной извлекатель для сложных PDF и офисных документов.",
            enabled = True,
            stage = "first-class",
            source_type = "python_package",
            package_name = "docling",
            module_name = "docling",
            installable = True,
            configurable = True,
            dependencies = ["docling"],
        ),
        RAGModuleCatalogItem(
            id = "tika",
            kind = "extractor",
            label = "Apache Tika",
            description = "Широкий fallback по форматам, включая базовый OCR-поток.",
            enabled = True,
            stage = "first-class",
            source_type = "python_package",
            package_name = "tika",
            module_name = "tika",
            installable = True,
            configurable = True,
            dependencies = ["tika", "java"],
        ),
        RAGModuleCatalogItem(
            id = "none",
            kind = "ocr",
            label = "No OCR",
            description = "Отключает OCR для профиля ingestion.",
            enabled = True,
            stage = "first-class",
            source_type = "builtin",
        ),
        RAGModuleCatalogItem(
            id = "rapidocr",
            kind = "ocr",
            label = "RapidOCR",
            description = "Быстрый OCR для Docling-профилей.",
            enabled = True,
            stage = "first-class",
            source_type = "python_package",
            package_name = "rapidocr-onnxruntime",
            module_name = "rapidocr_onnxruntime",
            installable = True,
            configurable = True,
            dependencies = ["rapidocr-onnxruntime"],
        ),
        RAGModuleCatalogItem(
            id = "tesseract",
            kind = "ocr",
            label = "Tesseract",
            description = "Широкий OCR fallback для документных сценариев.",
            enabled = True,
            stage = "first-class",
            source_type = "system_binary",
            binary_name = "tesseract",
            configurable = True,
            dependencies = ["tesseract"],
        ),
        RAGModuleCatalogItem(
            id = "bge-m3",
            kind = "embedder",
            label = "BGE-M3",
            description = "Рекомендуемый dense backend для первого этапа.",
            enabled = True,
            stage = "first-class",
            source_type = "local_model_path",
            package_name = "sentence-transformers",
            module_name = "sentence_transformers",
            default_model_id = "BAAI/bge-m3",
            installable = True,
            configurable = True,
            dependencies = ["sentence-transformers"],
            config_schema = {"model_id": "BAAI/bge-m3", "local_path": "/models/bge-m3"},
        ),
        RAGModuleCatalogItem(
            id = "multilingual-e5-large",
            kind = "embedder",
            label = "multilingual-e5-large",
            description = "Запасной мультиязычный профиль эмбеддингов.",
            enabled = True,
            stage = "first-class",
            source_type = "local_model_path",
            package_name = "sentence-transformers",
            module_name = "sentence_transformers",
            default_model_id = "intfloat/multilingual-e5-large",
            installable = True,
            configurable = True,
            dependencies = ["sentence-transformers"],
        ),
        RAGModuleCatalogItem(
            id = "none-reranker",
            kind = "reranker",
            label = "No Reranker",
            description = "Без второго этапа rerank.",
            enabled = True,
            stage = "first-class",
            source_type = "builtin",
        ),
        RAGModuleCatalogItem(
            id = "bge-reranker-v2-m3",
            kind = "reranker",
            label = "BGE Reranker v2 M3",
            description = "Опциональный reranker для более точного top-k.",
            enabled = True,
            stage = "first-class",
            source_type = "local_model_path",
            package_name = "sentence-transformers",
            module_name = "sentence_transformers",
            default_model_id = "BAAI/bge-reranker-v2-m3",
            installable = True,
            configurable = True,
            dependencies = ["sentence-transformers"],
        ),
        RAGModuleCatalogItem(
            id = "character-chunker",
            kind = "chunker",
            label = "Character Chunker",
            description = "Встроенный размерный chunking с overlap.",
            enabled = True,
            stage = "first-class",
            source_type = "builtin",
            configurable = True,
        ),
    ]
    return RAGModuleCatalogResponse(items = items)


def get_rag_module_by_id(module_id: str) -> RAGModuleCatalogItem | None:
    for item in get_rag_module_catalog().items:
        if item.id == module_id:
            return item
    if module_id == "none":
        return next((item for item in get_rag_module_catalog().items if item.id == "none-reranker"), None)
    return None


def _provider(item: RAGModuleCatalogItem, *, provider_id: str | None = None) -> RAGProviderOption:
    return RAGProviderOption(
        id = provider_id or item.id,
        label = item.label,
        description = item.description,
        enabled = item.enabled,
        stage = item.stage,
        kind = item.kind,
        source_type = item.source_type,
        package_name = item.package_name,
        module_name = item.module_name,
        binary_name = item.binary_name,
        default_model_id = item.default_model_id,
        installable = item.installable,
        configurable = item.configurable,
    )


def get_rag_providers() -> RAGProvidersResponse:
    catalog = get_rag_module_catalog().items
    extractor_items = [item for item in catalog if item.kind == "extractor" and item.id != "builtin-text"]
    extractor_items.extend(item for item in catalog if item.kind == "extractor" and item.id == "builtin-text")
    ocr_items = [item for item in catalog if item.kind == "ocr" and item.id != "none"]
    ocr_items.extend(item for item in catalog if item.kind == "ocr" and item.id == "none")
    return RAGProvidersResponse(
        vector_stores = [
            _provider(item)
            for item in catalog
            if item.kind == "vector_store"
        ],
        extractors = [
            _provider(item)
            for item in extractor_items
        ],
        ocr_engines = [
            _provider(item)
            for item in ocr_items
        ],
        embedders = [
            _provider(item)
            for item in catalog
            if item.kind == "embedder"
        ],
        rerankers = [
            _provider(item, provider_id = "none" if item.id == "none-reranker" else None)
            for item in catalog
            if item.kind == "reranker"
        ],
    )
