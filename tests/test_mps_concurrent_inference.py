# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Exercise the native Metal regression in a child process to contain SIGABRT."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from docling.utils.accelerator_utils import accelerator_inference_context


def test_rapid_ocr_reader_runs_inside_accelerator_guard(monkeypatch) -> None:
    import numpy
    from PIL import Image

    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

    active = False

    @contextmanager
    def guard(device):
        nonlocal active
        assert device == "mps"
        active = True
        try:
            yield
        finally:
            active = False

    def read(image, **kwargs):
        assert active
        return SimpleNamespace(
            boxes=numpy.zeros((1, 4, 2)), txts=["text"], scores=[0.9]
        )

    model = RapidOcrModel.__new__(RapidOcrModel)
    model._inference_device = "mps"
    model.options = SimpleNamespace(use_det=True, use_cls=True, use_rec=True)
    model.reader = read
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.accelerator_inference_context",
        guard,
    )
    assert model._read(Image.new("RGB", (4, 4))) == [([[0.0, 0.0]] * 4, "text", 0.9)]
    assert not active


@pytest.mark.parametrize("native_formula", [False, True])
def test_parsed_page_release_preserves_native_formula_input(native_formula) -> None:
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    pipeline = StandardPdfPipeline.__new__(StandardPdfPipeline)
    pipeline.pipeline_options = SimpleNamespace(
        generate_parsed_pages=False, do_native_formula_enrichment=native_formula
    )
    parsed = object()
    retained = object()
    pipeline.heading_hierarchy_model = Mock()
    pipeline.heading_hierarchy_model.retain_parsed_lines.return_value = retained
    page = SimpleNamespace(parsed_page=parsed)
    pipeline._release_parsed_page(page)
    assert page.parsed_page is (parsed if native_formula else retained)
    assert pipeline.heading_hierarchy_model.retain_parsed_lines.call_count == (
        0 if native_formula else 1
    )


def test_layout_guard_covers_transfer_and_materialization_only(monkeypatch) -> None:
    from PIL import Image

    from docling.models.inference_engines.object_detection.transformers_engine import (
        TransformersObjectDetectionEngine,
    )

    active = False
    events = []

    @contextmanager
    def guard(device):
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    class Inputs(dict):
        def to(self, device, dtype):
            assert active
            assert dtype == torch.float32
            events.append("transfer")
            return self

    def preprocess(**kwargs):
        assert not active
        events.append("prepare")
        return Inputs()

    def forward(**kwargs):
        assert active
        events.append("infer")

    def postprocess(*args, **kwargs):
        assert active
        events.append("postprocess")
        return [
            {
                "labels": torch.tensor([1]),
                "scores": torch.tensor([0.8]),
                "boxes": torch.zeros(1, 4),
            }
        ]

    tolist = torch.Tensor.tolist

    def materialize(tensor):
        assert active
        events.append("materialize")
        return tolist(tensor)

    def build(**kwargs):
        assert not active
        events.append("build")
        return kwargs

    engine = TransformersObjectDetectionEngine.__new__(
        TransformersObjectDetectionEngine
    )
    engine._device = torch.device("cpu")
    forward.dtype = torch.float32
    engine._model = forward
    engine._processor = Mock(side_effect=preprocess)
    engine._processor.post_process_object_detection = postprocess
    engine.options = Mock(score_threshold=0.5)
    engine._build_output = build
    monkeypatch.setattr(
        "docling.models.inference_engines.object_detection.transformers_engine.accelerator_inference_context",
        guard,
    )
    monkeypatch.setattr(torch.Tensor, "tolist", materialize)
    output = engine.predict_batch([Mock(image=Image.new("RGB", (4, 4)))])
    assert output[0]["labels"] == [1]
    assert events == [
        "prepare",
        "transfer",
        "infer",
        "postprocess",
        "materialize",
        "materialize",
        "materialize",
        "build",
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_non_mps_inference_remains_concurrent(device: str) -> None:
    barrier = Barrier(2)

    def submit() -> None:
        with accelerator_inference_context(device):
            barrier.wait(timeout=5)

    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(submit) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_concurrent_mps_batch_norm_and_arange_do_not_abort() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from docling.utils.accelerator_utils import accelerator_inference_context

torch.set_num_threads(4)
model = torch.nn.Sequential(
    torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.BatchNorm2d(64)
).eval().to('mps')
image = torch.randn(2, 64, 64, 64, device='mps')
# Complete initialization before concurrent inference begins.
reference = model(image).detach().cpu()
barrier = Barrier(2)

def batch_norm():
    barrier.wait(timeout=10)
    for _ in range(100):
        with accelerator_inference_context('mps'), torch.inference_mode():
            actual = model(image).cpu()
        torch.testing.assert_close(actual, reference)

def arange():
    barrier.wait(timeout=10)
    for _ in range(500):
        with accelerator_inference_context('mps'), torch.inference_mode():
            actual = torch.arange(400, device='mps').square().cpu()
        torch.testing.assert_close(actual, torch.arange(400).square())

with ThreadPoolExecutor(2) as executor:
    futures = [executor.submit(batch_norm), executor.submit(arange)]
    for future in futures:
        future.result(timeout=60)
""",
        ],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
