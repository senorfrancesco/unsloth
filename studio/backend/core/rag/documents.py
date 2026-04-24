# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from utils.paths import ensure_dir


ALLOWED_DOCUMENT_EXTS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024


@dataclass(frozen = True)
class ChunkFragment:
    text: str
    chunk_index: int
    start_index: int
    end_index: int


def normalize_text(raw: str) -> str:
    try:
        from data_designer_unstructured_seed.chunking import normalize_unstructured_text
    except ImportError:
        normalize_unstructured_text = None

    if normalize_unstructured_text is None:
        return raw.strip()
    return normalize_unstructured_text(raw)


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[ChunkFragment]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    step = max(chunk_size - chunk_overlap, 1)
    chunks: list[ChunkFragment] = []
    cursor = 0
    chunk_index = 0
    while cursor < len(normalized):
        raw_piece = normalized[cursor : cursor + chunk_size]
        piece = raw_piece.strip()
        if piece:
            leading_offset = len(raw_piece) - len(raw_piece.lstrip())
            start_index = cursor + leading_offset
            end_index = start_index + len(piece)
            chunks.append(
                ChunkFragment(
                    text = piece,
                    chunk_index = chunk_index,
                    start_index = start_index,
                    end_index = end_index,
                )
            )
            chunk_index += 1
        cursor += step
    return chunks


def read_extracted_text(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def write_text_document(
    *,
    dataset_dir: Path,
    document_name: str,
    text: str,
) -> dict[str, object]:
    document_id = uuid4().hex
    docs_dir = ensure_dir(dataset_dir / "documents")
    safe_name = Path(document_name).name or "document.txt"
    extracted_path = docs_dir / f"{document_id}.extracted.txt"
    normalized = normalize_text(text)
    extracted_path.write_text(normalized, encoding = "utf-8")
    return {
        "id": document_id,
        "document_name": safe_name,
        "mime_type": "text/plain",
        "size_bytes": len(text.encode("utf-8")),
        "text_char_count": len(normalized),
        "content_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "raw_path": None,
        "extracted_path": str(extracted_path),
        "metadata_path": None,
    }


async def store_uploaded_document(*, dataset_dir: Path, file: UploadFile) -> dict[str, object]:
    original_filename = Path(file.filename or "upload").name
    ext = Path(original_filename).suffix.lower()
    if ext not in ALLOWED_DOCUMENT_EXTS:
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code = 400, detail = "Empty file not allowed.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code = 413, detail = "File too large.")

    document_id = uuid4().hex
    docs_dir = ensure_dir(dataset_dir / "documents")
    raw_path = docs_dir / f"{document_id}{ext}"
    raw_path.write_bytes(content)
    extracted_text = _extract_text_from_file(raw_path, ext)
    if not extracted_text.strip():
        raw_path.unlink(missing_ok = True)
        raise HTTPException(status_code = 422, detail = "No extractable text found in file.")
    extracted_path = docs_dir / f"{document_id}.extracted.txt"
    extracted_path.write_text(extracted_text, encoding = "utf-8")
    metadata_path = docs_dir / f"{document_id}.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "original_filename": original_filename,
                "size_bytes": len(content),
                "content_type": file.content_type or "application/octet-stream",
            },
            ensure_ascii = False,
        ),
        encoding = "utf-8",
    )
    return {
        "id": document_id,
        "document_name": original_filename,
        "mime_type": file.content_type or "application/octet-stream",
        "size_bytes": len(content),
        "text_char_count": len(extracted_text),
        "content_hash": hashlib.sha256(extracted_text.encode("utf-8")).hexdigest(),
        "raw_path": str(raw_path),
        "extracted_path": str(extracted_path),
        "metadata_path": str(metadata_path),
    }


def build_qdrant_points(
    *,
    collection_id: str,
    projection_id: str,
    documents: Iterable[dict[str, object]],
    chunk_size: int,
    chunk_overlap: int,
    extractor: str,
    ocr_engine: str,
    chunk_recipe: str,
    embedding_config: dict[str, object],
    vectors: list[list[float]],
) -> tuple[list[dict[str, object]], int]:
    points: list[dict[str, object]] = []
    vector_cursor = 0
    total_chunks = 0
    for document in documents:
        extracted_path = Path(str(document["extracted_path"]))
        chunks = chunk_text(
            read_extracted_text(extracted_path),
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )
        for chunk in chunks:
            vector = vectors[vector_cursor]
            vector_cursor += 1
            total_chunks += 1
            point_id = hashlib.sha256(
                f"{projection_id}:{document['id']}:{chunk.chunk_index}".encode("utf-8")
            ).hexdigest()
            points.append(
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "collection_id": collection_id,
                        "knowledge_base_id": collection_id,
                        "dataset_id": document["dataset_id"],
                        "document_id": document["id"],
                        "file_id": document["id"],
                        "knowledge_file_id": document["id"],
                        "document_name": document["document_name"],
                        "name": document["document_name"],
                        "source": document["document_name"],
                        "created_by": None,
                        "content_type": document["mime_type"],
                        "chunk_index": chunk.chunk_index,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "page": None,
                        "section_path": None,
                        "language": None,
                        "text": chunk.text,
                        "source_scope": "knowledge",
                        "extractor": extractor,
                        "ocr_engine": ocr_engine,
                        "chunk_recipe": chunk_recipe,
                        "embedding_config": embedding_config,
                        "source_hash": document["content_hash"],
                        "hash": document["content_hash"],
                        "projection_id": projection_id,
                    },
                }
            )
    return points, total_chunks


def _extract_text_from_file(file_path: Path, ext: str) -> str:
    if ext in {".txt", ".md"}:
        raw = file_path.read_text(encoding = "utf-8", errors = "ignore")
        return normalize_text(raw)
    if ext == ".pdf":
        try:
            import pymupdf4llm
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("PDF extraction dependency `pymupdf4llm` is missing.") from exc
        raw = pymupdf4llm.to_markdown(
            str(file_path), write_images = False, show_progress = False, use_ocr = False
        )
        return normalize_text(raw)
    if ext == ".docx":
        try:
            import mammoth
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("DOCX extraction dependency `mammoth` is missing.") from exc
        with open(file_path, "rb") as handle:
            raw = mammoth.convert_to_markdown(handle).value
        return normalize_text(raw)
    raise RuntimeError(f"Unsupported file type: {ext}")
