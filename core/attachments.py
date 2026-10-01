"""Multimodal attachment processor for images, PDFs, code, and text documents."""

import base64
import io
import mimetypes
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import zlib

import structlog

logger = structlog.get_logger()

# Common code & text file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".html", ".htm", ".css", ".scss", ".sass", ".sh", ".bash", ".zsh",
    ".sql", ".xml", ".svg", ".log", ".env", ".dockerfile", ".r", ".c",
    ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".kt", ".swift",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".ico", ".tiff",
}


def clean_base64(data_str: str) -> tuple[str, str]:
    """Parse a data URI or raw base64 string, returning (clean_base64, mime_type)."""
    if not data_str:
        return "", ""
    data_str = data_str.strip()
    if data_str.startswith("data:"):
        prefix, _, b64_part = data_str.partition(",")
        mime = prefix.split(";")[0].replace("data:", "").strip().lower()
        return b64_part.strip(), mime
    return data_str, "application/octet-stream"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdftotext or pure-Python FlateDecode fallback."""
    # 1. Primary: Use poppler pdftotext utility if installed
    try:
        proc = subprocess.Popen(
            ["pdftotext", "-layout", "-", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = proc.communicate(input=pdf_bytes, timeout=10)
        if proc.returncode == 0 and stdout:
            text = stdout.decode("utf-8", errors="replace").strip()
            if text:
                return text
    except Exception as e:
        logger.debug("pdftotext_cli_failed", error=str(e))

    # 2. Secondary fallback: Pure-Python stream decompression
    try:
        extracted = []
        # Find all stream ... endstream blocks
        stream_matches = re.finditer(b"stream[\r\n]+(.*?)[\r\n]+endstream", pdf_bytes, re.DOTALL)
        for m in stream_matches:
            raw_stream = m.group(1)
            # Try FlateDecode
            decompressed = None
            try:
                decompressed = zlib.decompress(raw_stream)
            except Exception:
                try:
                    decompressed = zlib.decompress(raw_stream, -zlib.MAX_WBITS)
                except Exception:
                    decompressed = raw_stream

            if decompressed:
                # Extract text inside parenthesis e.g. (Hello World) Tj or [(Hello) (World)] TJ
                text_matches = re.findall(rb"\((.*?)\)\s*T[jJ]", decompressed)
                for tm in text_matches:
                    try:
                        extracted.append(tm.decode("latin1", errors="ignore"))
                    except Exception:
                        pass
        joined = " ".join(extracted).strip()
        if joined:
            return joined
    except Exception as e:
        logger.warning("pure_python_pdf_extract_failed", error=str(e))

    return "[PDF text could not be extracted; binary or scanned format]"


def process_attachment(item: dict[str, Any]) -> dict[str, Any]:
    """Process a single attachment item into standardized format."""
    name = str(item.get("name") or "attachment").strip()
    raw_type = str(item.get("type") or "").strip().lower()
    raw_data = str(item.get("data") or "")
    size = int(item.get("size") or len(raw_data))

    ext = Path(name).suffix.lower()

    # Determine MIME type if missing
    if not raw_type or raw_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(name)
        if guessed:
            raw_type = guessed.lower()

    b64_content, detected_mime = clean_base64(raw_data)
    effective_mime = detected_mime if detected_mime != "application/octet-stream" else raw_type

    # 1. Images
    if effective_mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return {
            "kind": "image",
            "name": name,
            "mime_type": effective_mime if effective_mime.startswith("image/") else "image/png",
            "base64": b64_content,
            "size": size,
        }

    # 2. PDFs
    if effective_mime == "application/pdf" or ext == ".pdf":
        try:
            pdf_bytes = base64.b64decode(b64_content)
            extracted_text = extract_pdf_text(pdf_bytes)
        except Exception as e:
            extracted_text = f"[Failed to decode PDF: {str(e)}]"

        return {
            "kind": "pdf",
            "name": name,
            "mime_type": "application/pdf",
            "text": extracted_text,
            "size": size,
        }

    # 3. Code & Text Files
    if (
        effective_mime.startswith("text/")
        or effective_mime in ("application/json", "application/javascript", "application/xml", "application/x-sh")
        or ext in TEXT_EXTENSIONS
    ):
        text_content = ""
        # Try base64 decode first
        try:
            decoded_bytes = base64.b64decode(b64_content)
            text_content = decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            # If not valid base64, raw_data is likely already plain text
            text_content = raw_data

        return {
            "kind": "text",
            "name": name,
            "mime_type": effective_mime or "text/plain",
            "text": text_content,
            "size": size,
        }

    # 4. Fallback binary/unsupported
    return {
        "kind": "binary",
        "name": name,
        "mime_type": effective_mime,
        "size": size,
    }


def compose_turn_prompt_with_attachments(
    user_query: str,
    attachments: list[dict[str, Any]] | None,
) -> tuple[str, list[str]]:
    """Compose user prompt injecting document contents and extracting base64 images.

    Returns:
        (composed_prompt_string, list_of_image_base64_strings)
    """
    if not attachments:
        return user_query, []

    processed = [process_attachment(a) for a in attachments if isinstance(a, dict)]

    image_base64_list: list[str] = []
    doc_sections: list[str] = []

    for item in processed:
        kind = item.get("kind")
        name = item.get("name", "attachment")

        if kind == "image":
            b64 = item.get("base64")
            if b64:
                image_base64_list.append(b64)
        elif kind == "pdf":
            text = item.get("text", "").strip()
            doc_sections.append(
                f"### [Attached Document: {name} (PDF)]\n"
                f"```text\n{text}\n```\n"
            )
        elif kind == "text":
            text = item.get("text", "").strip()
            ext = Path(name).suffix.lstrip(".").lower() or "text"
            doc_sections.append(
                f"### [Attached File: {name}]\n"
                f"```{ext}\n{text}\n```\n"
            )
        elif kind == "binary":
            size_kb = max(1, item.get("size", 0) // 1024)
            doc_sections.append(
                f"### [Attached Binary File: {name} ({size_kb} KB)]\n"
                f"(Binary file attached by user)\n"
            )

    composed_query = user_query.strip()
    if doc_sections:
        attachment_block = "\n".join(doc_sections)
        composed_query = f"{composed_query}\n\n{attachment_block}".strip()

    return composed_query, image_base64_list
