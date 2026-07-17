"""HTTP API for PDF translation service using FastAPI.

Stateless design: one streaming request per translation. The caller passes a
signed GET URL for the source PDF and signed PUT URLs for the results; this
service pulls the source, translates, pushes the results directly to storage,
and streams progress back over SSE on the same connection.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chardet
import httpx

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field
from sse_starlette.sse import EventSourceResponse

from pdf2zh_next.high_level import do_translate_async_stream
from pdf2zh_next.models_config import SUPPORTED_MODELS, DEFAULT_MODEL, is_model_supported

logger = logging.getLogger(__name__)

# Generous transfer timeout for large PDFs (source pull / result push).
_TRANSFER_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0)


class TranslationRequest(BaseModel):
    """Request model for translation."""

    # Model selection (required)
    model: str = Field(
        default=DEFAULT_MODEL,
        description=f"Translation model name. Use GET /v1/models to see all available models. Default: {DEFAULT_MODEL}"
    )

    # Basic settings
    lang_in: str = Field(default="en", description="Source language code")
    lang_out: str = Field(default="zh", description="Target language code")
    pages: str | None = Field(
        default=None, description="Pages to translate (e.g. '1,2,1-,-3,3-5')"
    )
    no_dual: bool = Field(
        default=False, description="Do not output bilingual PDF files"
    )
    no_mono: bool = Field(
        default=False, description="Do not output monolingual PDF files"
    )

    # PDF processing options
    split_short_lines: bool = Field(
        default=False, description="Force split short lines into different paragraphs"
    )
    short_line_split_factor: float = Field(
        default=0.8, description="Split threshold factor for short lines"
    )
    skip_clean: bool = Field(
        default=False, description="Skip PDF cleaning step"
    )
    dual_translate_first: bool = Field(
        default=False, description="Put translated pages first in dual PDF mode"
    )
    disable_rich_text_translate: bool = Field(
        default=False, description="Disable rich text translation"
    )
    enhance_compatibility: bool = Field(
        default=False, description="Enable all compatibility enhancement options"
    )
    use_alternating_pages_dual: bool = Field(
        default=False, description="Use alternating pages mode for dual PDF"
    )
    watermark_output_mode: str = Field(
        default="no_watermark",
        description="Control watermark output mode: 'watermarked' (default), 'no_watermark', or 'both'"
    )
    max_pages_per_part: int | None = Field(
        default=None, description="Maximum number of pages per part for split translation"
    )
    translate_table_text: bool = Field(
        default=True, description="Translate table text (experimental)"
    )
    formular_font_pattern: str | None = Field(
        default=None, description="Font pattern to identify formula text"
    )
    formular_char_pattern: str | None = Field(
        default=None, description="Character pattern to identify formula text"
    )
    skip_scanned_detection: bool = Field(
        default=False, description="Skip scanned document detection"
    )
    ocr_workaround: bool = Field(
        default=False, description="Use OCR workaround for scanned PDFs"
    )
    auto_enable_ocr_workaround: bool = Field(
        default=False, description="Enable automatic OCR workaround for scanned PDFs"
    )
    only_include_translated_page: bool = Field(
        default=False, description="Only include translated pages in the output PDF"
    )
    merge_alternating_line_numbers: bool = Field(
        default=True, description="Enable merging alternating line-number layouts"
    )
    remove_non_formula_lines: bool = Field(
        default=True, description="Remove non-formula lines from paragraph areas"
    )
    non_formula_line_iou_threshold: float = Field(
        default=0.9, description="IoU threshold for detecting paragraph overlap"
    )
    figure_table_protection_threshold: float = Field(
        default=0.9, description="IoU threshold for protecting lines in figure/table areas"
    )
    primary_font_family: str | None = Field(
        default=None, description="Override primary font family: 'serif', 'sans-serif', or 'script'"
    )

    # Translation service options
    min_text_length: int = Field(
        default=5, description="Minimum text length to translate"
    )
    qps: int = Field(
        default=4, description="QPS (Queries Per Second) limit for translation service"
    )
    ignore_cache: bool = Field(
        default=False, description="Ignore translation cache and force retranslation"
    )
    custom_system_prompt: str | None = Field(
        default=None, description="Custom system prompt for translation"
    )
    pool_max_workers: int | None = Field(
        default=None, description="Maximum number of worker threads for task processing"
    )

    # Glossary options
    no_auto_extract_glossary: bool = Field(
        default=True, description="Disable automatic term extraction"
    )
    save_auto_extracted_glossary: bool = Field(
        default=False, description="Save automatically extracted glossary"
    )

    # Advanced options
    rpc_doclayout: str | None = Field(
        default=None, description="RPC service host address for document layout analysis"
    )

    # OpenAI translator specific
    openai_model: str | None = Field(default=None, description="OpenAI model")
    openai_base_url: str | None = Field(default=None, description="OpenAI base URL")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")


class GlossaryRef(BaseModel):
    """A glossary CSV referenced by a signed GET URL."""

    url: str = Field(description="Signed GET URL to download the glossary CSV")
    name: str = Field(description="Original file name (used for error messages)")


class StreamTranslationRequest(TranslationRequest):
    """Streaming translation request with signed-URL direct transfer."""

    source_url: str = Field(description="Signed GET URL of the source PDF")
    mono_put_url: str | None = Field(
        default=None, description="Signed PUT URL for the mono result; null → skip mono"
    )
    dual_put_url: str | None = Field(
        default=None, description="Signed PUT URL for the dual result; null → skip dual"
    )
    glossaries: list[GlossaryRef] = Field(
        default_factory=list, description="Glossary CSVs referenced by signed GET URLs"
    )


def validate_glossary_csv(content: str, filename: str) -> tuple[bool, str | None]:
    """Validate that the CSV content has required columns.

    Args:
        content: CSV file content as string
        filename: Original filename for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Try to parse CSV
        reader = csv.DictReader(io.StringIO(content))
        fieldnames = reader.fieldnames

        if not fieldnames:
            return False, "CSV file is empty or has no header row"

        # Check for required columns
        required_columns = {"source", "target"}
        available_columns = {col.lower().strip() for col in fieldnames}

        missing_columns = required_columns - available_columns
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}. CSV must have 'source' and 'target' columns"

        # Check if there's at least one data row
        first_row = next(reader, None)
        if first_row is None:
            return False, "CSV file has header but no data rows"

        return True, None

    except csv.Error as e:
        return False, f"Invalid CSV format: {e}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for the FastAPI app."""
    logger.info("HTTP API server starting...")

    # Disable noisy loggers
    for logger_name in ["httpx", "openai", "httpcore", "http11"]:
        logging.getLogger(logger_name).setLevel("CRITICAL")
        logging.getLogger(logger_name).propagate = False

    yield

    logger.info("HTTP API server shutting down...")


app = FastAPI(
    title="PDF Translation API",
    description="Stateless API for translating PDF documents via signed-URL direct transfer",
    version="2.6.4",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


def create_settings_from_request(
    request: TranslationRequest, input_file: Path, output_dir: Path,
    glossaries: str | None = None
) -> Any:
    """Create settings from translation request.

    All translations use OpenAI Compatible API.
    Credentials are resolved per request, falling back to environment variables:
    - base URL: request.openai_base_url or OPENAI_API_BASE
    - API key:  request.openai_api_key or OPENAI_API_KEY

    Args:
        request: Translation request parameters
        input_file: Path to input PDF file
        output_dir: Directory for output files
        glossaries: Comma-separated list of glossary file paths
    """
    import os
    from pdf2zh_next.config import (
        BasicSettings,
        OpenAICompatibleSettings,
        PDFSettings,
        SettingsModel,
        TranslationSettings,
    )

    # Resolve configuration: per-request values take precedence over env vars
    base_url = request.openai_base_url or os.environ.get("OPENAI_API_BASE")
    api_key = request.openai_api_key or os.environ.get("OPENAI_API_KEY")

    # A per-request base URL routes through an upstream gateway that owns the
    # authoritative model catalog, so defer model validation to it; env-var-only
    # callers keep the local whitelist check.
    if request.openai_base_url:
        logger.info(
            f"Per-request base URL set; deferring model validation for '{request.model}' to upstream"
        )
    elif not is_model_supported(request.model):
        raise ValueError(
            f"Unsupported model: {request.model}. "
            f"Use GET /v1/models to see all available models."
        )

    if not base_url:
        raise ValueError(
            "OpenAI-compatible base URL is required. "
            "Provide 'openai_base_url' in the request or set the "
            "OPENAI_API_BASE environment variable."
        )
    if not api_key:
        raise ValueError(
            "OpenAI-compatible API key is required. "
            "Provide 'openai_api_key' in the request or set the "
            "OPENAI_API_KEY environment variable."
        )

    logger.info(f"Using model: {request.model}")

    # Create basic settings
    basic = BasicSettings(input_files={str(input_file)})

    # Log translation parameters for debugging
    logger.info(f"Translation parameters: model={request.model}, lang_in={request.lang_in}, lang_out={request.lang_out}, "
                f"qps={request.qps}, min_text_length={request.min_text_length}, ignore_cache={request.ignore_cache}")

    # Create translation settings with output directory
    translation = TranslationSettings(
        lang_in=request.lang_in,
        lang_out=request.lang_out,
        min_text_length=request.min_text_length,
        qps=request.qps,
        ignore_cache=request.ignore_cache,
        custom_system_prompt=request.custom_system_prompt,
        pool_max_workers=request.pool_max_workers,
        no_auto_extract_glossary=request.no_auto_extract_glossary,
        save_auto_extracted_glossary=request.save_auto_extracted_glossary,
        primary_font_family=request.primary_font_family,
        rpc_doclayout=request.rpc_doclayout,
        output=str(output_dir),  # Set output directory
        glossaries=glossaries,  # Pass glossary file paths
    )

    # Create PDF settings
    pdf = PDFSettings(
        pages=request.pages,
        no_dual=request.no_dual,
        no_mono=request.no_mono,
        split_short_lines=request.split_short_lines,
        short_line_split_factor=request.short_line_split_factor,
        skip_clean=request.skip_clean,
        dual_translate_first=request.dual_translate_first,
        disable_rich_text_translate=request.disable_rich_text_translate,
        enhance_compatibility=request.enhance_compatibility,
        use_alternating_pages_dual=request.use_alternating_pages_dual,
        watermark_output_mode=request.watermark_output_mode,
        max_pages_per_part=request.max_pages_per_part,
        translate_table_text=request.translate_table_text,
        formular_font_pattern=request.formular_font_pattern,
        formular_char_pattern=request.formular_char_pattern,
        skip_scanned_detection=request.skip_scanned_detection,
        ocr_workaround=request.ocr_workaround,
        auto_enable_ocr_workaround=request.auto_enable_ocr_workaround,
        only_include_translated_page=request.only_include_translated_page,
        no_merge_alternating_line_numbers=not request.merge_alternating_line_numbers,
        no_remove_non_formula_lines=not request.remove_non_formula_lines,
        non_formula_line_iou_threshold=request.non_formula_line_iou_threshold,
        figure_table_protection_threshold=request.figure_table_protection_threshold,
    )

    # Create OpenAI Compatible settings
    translate_engine_settings = OpenAICompatibleSettings(
        openai_compatible_model=request.model,
        openai_compatible_base_url=base_url,
        openai_compatible_api_key=api_key,
    )

    # Create settings model
    settings = SettingsModel(
        basic=basic,
        translation=translation,
        pdf=pdf,
        translate_engine_settings=translate_engine_settings,
    )

    return settings


async def _download_to_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Stream a signed GET URL to a local file."""
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            async for chunk in resp.aiter_bytes():
                f.write(chunk)


async def _upload_file(client: httpx.AsyncClient, put_url: str, src: Path) -> None:
    """Upload a local file to a signed PUT URL.

    No Content-Type header is sent: the presigned PUT URL is signed with an empty
    content-type, so the request must match to avoid a signature mismatch.
    """
    with open(src, "rb") as f:
        data = f.read()
    resp = await client.put(put_url, content=data)
    resp.raise_for_status()


async def _download_glossaries(
    client: httpx.AsyncClient, refs: list[GlossaryRef], work_dir: Path
) -> str | None:
    """Download, validate, and persist glossary CSVs; return comma-joined paths."""
    if not refs:
        return None
    paths: list[str] = []
    for idx, ref in enumerate(refs):
        resp = await client.get(ref.url)
        resp.raise_for_status()
        content = resp.content
        detected = chardet.detect(content)
        encoding = detected.get("encoding") or "utf-8"
        text = content.decode(encoding, errors="replace")

        is_valid, error_msg = validate_glossary_csv(text, ref.name)
        if not is_valid:
            raise ValueError(f"Glossary '{ref.name}': {error_msg}")

        safe_name = ref.name.replace("/", "_")
        path = work_dir / f"glossary_{idx}_{safe_name}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(str(path))
    return ",".join(paths) if paths else None


@app.post("/v1/translate/stream")
async def create_translation_stream(request: StreamTranslationRequest):
    """Run one translation over a single streaming connection.

    Pulls the source PDF from ``source_url``, translates, pushes the mono/dual
    results to their signed PUT URLs, and streams progress as SSE events:

        event: progress   data: {"percentage", "stage", "message"}
        event: done        data: {"mono": bool, "dual": bool, "time_cost_seconds"}
        event: error       data: {"message"}
    """
    # Skip an output when no PUT target was provided for it.
    request.no_mono = request.no_mono or request.mono_put_url is None
    request.no_dual = request.no_dual or request.dual_put_url is None

    work_dir = Path(tempfile.mkdtemp(prefix="pdf2zh_stream_"))

    async def event_generator():
        try:
            input_file = work_dir / "input.pdf"
            async with httpx.AsyncClient(
                timeout=_TRANSFER_TIMEOUT, follow_redirects=True
            ) as client:
                await _download_to_file(client, request.source_url, input_file)
                glossaries = await _download_glossaries(client, request.glossaries, work_dir)

            settings = create_settings_from_request(request, input_file, work_dir, glossaries)

            async for event in do_translate_async_stream(settings, input_file):
                event_type = event.get("type", "unknown")

                if event_type in ("progress_start", "progress_update", "progress_end"):
                    stage = event.get("stage", "Processing")
                    overall_progress = event.get("overall_progress", 0)
                    stage_current = event.get("stage_current", 0)
                    stage_total = event.get("stage_total", 0)
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "percentage": round(overall_progress, 2),
                            "stage": stage,
                            "message": f"{stage}: {stage_current}/{stage_total}",
                        }),
                    }

                elif event_type == "finish":
                    result = event["translate_result"]
                    mono_done = False
                    dual_done = False
                    async with httpx.AsyncClient(
                        timeout=_TRANSFER_TIMEOUT, follow_redirects=True
                    ) as up:
                        if result.mono_pdf_path and request.mono_put_url:
                            await _upload_file(up, request.mono_put_url, Path(result.mono_pdf_path))
                            mono_done = True
                        if result.dual_pdf_path and request.dual_put_url:
                            await _upload_file(up, request.dual_put_url, Path(result.dual_pdf_path))
                            dual_done = True

                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "percentage": 100.0,
                            "stage": "Completed",
                            "message": "Translation completed",
                        }),
                    }
                    yield {
                        "event": "done",
                        "data": json.dumps({
                            "mono": mono_done,
                            "dual": dual_done,
                            "time_cost_seconds": result.total_seconds,
                        }),
                    }
                    logger.info("Streaming translation completed successfully")
                    return

                elif event_type == "error":
                    message = event.get("error", "Unknown error")
                    logger.error(f"Translation error: {message}")
                    yield {"event": "error", "data": json.dumps({"message": message})}
                    return

        except asyncio.CancelledError:
            # Client disconnect / shutdown: propagate so do_translate_async_stream
            # kills its subprocess instead of leaving a zombie translation.
            logger.info("Streaming translation cancelled (client disconnect)")
            raise
        except Exception as e:
            logger.error(f"Streaming translation failed: {e}")
            yield {"event": "error", "data": json.dumps({"message": str(e)})}
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    return EventSourceResponse(event_generator())


@app.get("/v1/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pdf2zh-next-api"}


@app.get("/v1/models")
async def list_models():
    """Get list of supported translation models."""
    return {
        "models": SUPPORTED_MODELS,
        "total": len(SUPPORTED_MODELS),
        "default_model": DEFAULT_MODEL,
    }


def run_server(host: str = "0.0.0.0", port: int = 11008, reload: bool = False):
    """Run the HTTP API server."""
    import uvicorn

    uvicorn.run(
        "pdf2zh_next.http_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True,
        log_config=None,  # let uvicorn use root logger (Rich)
    )


if __name__ == "__main__":
    run_server()
