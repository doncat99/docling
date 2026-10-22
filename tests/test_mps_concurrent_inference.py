# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Exercise the native Metal regression in a child process to contain SIGABRT."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from unittest.mock import Mock

import pytest
import torch

from docling.utils.accelerator_utils import accelerator_inference_context


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
        def to(self, device):
            assert active
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
