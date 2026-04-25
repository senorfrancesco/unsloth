# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import gzip
import hashlib
import io
import json
import mimetypes
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from utils.paths import ensure_dir


ALLOWED_DOCUMENT_EXTS = {".txt", ".md", ".pdf", ".docx"}
ARCHIVE_EXTS = {".zip", ".tar", ".tar.gz", ".tgz", ".gz", ".rar"}
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ARCHIVE_DOCUMENTS = 500


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
    documents = await store_uploaded_documents(dataset_dir = dataset_dir, file = file)
    return documents[0]


async def store_uploaded_documents(*, dataset_dir: Path, file: UploadFile) -> list[dict[str, object]]:
    original_filename = _safe_archive_member_name(file.filename or "upload")
    ext = _document_ext(original_filename)
    archive_ext = _archive_ext(original_filename)
    if ext not in ALLOWED_DOCUMENT_EXTS and archive_ext not in ARCHIVE_EXTS:
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported file type: {ext}",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code = 400, detail = "Empty file not allowed.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code = 413, detail = "File too large.")

    if archive_ext in ARCHIVE_EXTS:
        documents = _store_archive_documents(
            dataset_dir = dataset_dir,
            archive_filename = original_filename,
            content = content,
            content_type = file.content_type or "application/octet-stream",
            archive_ext = archive_ext,
        )
        if not documents:
            raise HTTPException(status_code = 422, detail = "No supported documents found in archive.")
        return documents

    return [
        _store_document_content(
            dataset_dir = dataset_dir,
            document_name = original_filename,
            content = content,
            content_type = file.content_type or mimetypes.guess_type(original_filename)[0] or "application/octet-stream",
            archive_filename = None,
            archive_member = None,
        )
    ]


def _store_document_content(
    *,
    dataset_dir: Path,
    document_name: str,
    content: bytes,
    content_type: str,
    archive_filename: str | None,
    archive_member: str | None,
) -> dict[str, object]:
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code = 413, detail = f"File `{document_name}` is too large.")
    ext = _document_ext(document_name)
    if ext not in ALLOWED_DOCUMENT_EXTS:
        raise HTTPException(status_code = 400, detail = f"Unsupported file type: {ext}")

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
                "original_filename": document_name,
                "size_bytes": len(content),
                "content_type": content_type,
                "archive_filename": archive_filename,
                "archive_member": archive_member,
            },
            ensure_ascii = False,
        ),
        encoding = "utf-8",
    )
    return {
        "id": document_id,
        "document_name": document_name,
        "mime_type": content_type,
        "size_bytes": len(content),
        "text_char_count": len(extracted_text),
        "content_hash": hashlib.sha256(extracted_text.encode("utf-8")).hexdigest(),
        "raw_path": str(raw_path),
        "extracted_path": str(extracted_path),
        "metadata_path": str(metadata_path),
    }


def _store_archive_documents(
    *,
    dataset_dir: Path,
    archive_filename: str,
    content: bytes,
    content_type: str,
    archive_ext: str,
) -> list[dict[str, object]]:
    documents: list[dict[str, object]] = []
    for member_name, member_content in _iter_archive_members(
        archive_filename = archive_filename,
        content = content,
        archive_ext = archive_ext,
    ):
        if len(documents) >= MAX_ARCHIVE_DOCUMENTS:
            raise HTTPException(status_code = 413, detail = "Archive contains too many documents.")
        safe_name = _safe_archive_member_name(member_name)
        if _document_ext(safe_name) not in ALLOWED_DOCUMENT_EXTS:
            continue
        documents.append(
            _store_document_content(
                dataset_dir = dataset_dir,
                document_name = safe_name,
                content = member_content,
                content_type = mimetypes.guess_type(safe_name)[0] or content_type,
                archive_filename = archive_filename,
                archive_member = member_name,
            )
        )
    return documents


def _iter_archive_members(
    *,
    archive_filename: str,
    content: bytes,
    archive_ext: str,
) -> Iterable[tuple[str, bytes]]:
    if archive_ext == ".zip":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                yield item.filename, archive.read(item)
        return
    if archive_ext in {".tar", ".tar.gz", ".tgz"}:
        with tarfile.open(fileobj = io.BytesIO(content), mode = "r:*") as archive:
            for item in archive.getmembers():
                if not item.isfile():
                    continue
                handle = archive.extractfile(item)
                if handle is None:
                    continue
                yield item.name, handle.read()
        return
    if archive_ext == ".gz":
        inner_name = archive_filename.removesuffix(".gz")
        if _document_ext(inner_name) not in ALLOWED_DOCUMENT_EXTS:
            inner_name = f"{Path(inner_name).name or 'document'}.txt"
        yield inner_name, gzip.decompress(content)
        return
    if archive_ext == ".rar":
        try:
            import rarfile
        except ImportError as exc:  # pragma: no cover - depends on runtime
            raise RuntimeError("RAR extraction dependency `rarfile` is missing.") from exc
        with rarfile.RarFile(io.BytesIO(content)) as archive:  # pragma: no cover - depends on runtime
            for item in archive.infolist():
                if item.isdir():
                    continue
                yield item.filename, archive.read(item)
        return
    raise RuntimeError(f"Unsupported archive type: {archive_ext}")


def _safe_archive_member_name(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    parts = [
        part
        for part in path.parts
        if part not in {"", ".", ".."} and not part.startswith("/")
    ]
    return "/".join(parts) or Path(value).name or "document.txt"


def _document_ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _archive_ext(filename: str) -> str:
    value = filename.lower()
    if value.endswith(".tar.gz"):
        return ".tar.gz"
    if value.endswith(".tgz"):
        return ".tgz"
    return Path(value).suffix.lower()


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
    indexed_at: str | None = None,
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
                        "indexed_at": indexed_at,
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
