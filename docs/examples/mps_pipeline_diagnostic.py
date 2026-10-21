# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Diagnose concurrent MPS conversion without injecting device synchronization."""

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.backend_options import PdfBackendOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "output", type=Path, help="New evidence directory; must not exist"
    )
    parser.add_argument("--artifacts-path", type=Path)
    args = parser.parse_args()
    if not args.pdf.is_file():
        parser.error("PDF must be an existing file")
    if not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable PYTORCH_ENABLE_MPS_FALLBACK for this diagnostic")
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(4)
    settings.debug.profile_pipeline_timings = True
    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device="mps", num_threads=4),
        artifacts_path=args.artifacts_path,
        do_ocr=False,
        generate_parsed_pages=True,
        do_native_formula_enrichment=True,
    )
    options.layout_options.recover_formula_regions = True
    metadata = {
        "torch": torch.__version__,
        "docling": version("docling"),
        "docling_ibm_models": version("docling-ibm-models"),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "pdf_sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "explicit_sync": False,
        "options": options.model_dump(mode="json"),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    start = time.perf_counter()
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=options,
                backend_options=PdfBackendOptions(enforce_same_font=False),
            )
        }
    )
    # Keep the normal threaded pipeline. A device-wide barrier inside a model
    # callback changes its execution and can trigger the Metal assertion.
    result = converter.convert(args.pdf)
    elapsed = time.perf_counter() - start
    result.document.save_as_json(args.output / "document.json")
    summary = {
        "seconds": elapsed,
        "timing_semantics": "conversion wall time including model initialization",
        "status": result.status.value,
        "pages": len(result.pages),
        "tables": len(result.document.tables),
        "cells": sum(len(table.data.table_cells) for table in result.document.tables),
        "errors": [str(error) for error in result.errors],
        "timings": {key: value.times for key, value in result.timings.items()},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logging.info("Conversion result: %s", summary)
    if result.status != ConversionStatus.SUCCESS or result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
