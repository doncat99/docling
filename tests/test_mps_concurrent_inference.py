# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Exercise the native Metal regression in a child process to contain SIGABRT."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch

from docling.utils.accelerator_utils import accelerator_inference_context


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
