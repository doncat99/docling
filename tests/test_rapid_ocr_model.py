# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

from io import BytesIO
from pathlib import Path

import pytest

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

pytestmark = pytest.mark.ml_ocr


def _capture_params(
    monkeypatch: pytest.MonkeyPatch,
    options: RapidOcrOptions,
    artifacts_path: Path,
    resolved_device: str = "cpu",
    requested_device: str = "cpu",
    available_providers: list[str] | None = None,
) -> dict[str, object]:
    """Build a RapidOcrModel with real rapidocr resolution but faked inference
    and downloading, returning the params dict handed to RapidOCR.

    artifacts_path is strictly offline, so the checkpoints are prefetched first.
    """
    import onnxruntime
    import rapidocr

    if available_providers is not None:
        monkeypatch.setattr(
            onnxruntime, "get_available_providers", lambda: available_providers
        )
    elif resolved_device.startswith("cuda"):
        monkeypatch.setattr(
            onnxruntime,
            "get_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    captured: dict[str, object] = {}

    class FakeRapidOCR:
        def __init__(self, *, params):
            captured["params"] = params

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.decide_device",
        lambda device: resolved_device,
    )
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.download_url_with_progress",
        lambda url, *, progress: BytesIO(b"dummy content"),
    )
    RapidOcrModel.download_models(
        backend=options.backend,
        lang=options.lang[0],
        local_dir=artifacts_path / RapidOcrModel._model_repo_folder,
    )

    RapidOcrModel(
        enabled=True,
        artifacts_path=artifacts_path,
        options=options,
        accelerator_options=AcceleratorOptions(device=requested_device, num_threads=4),
    )
    return captured["params"]


@pytest.mark.parametrize(
    ("backend", "engine_key"),
    [
        ("onnxruntime", "EngineConfig.onnxruntime.intra_op_num_threads"),
        ("openvino", "EngineConfig.openvino.inference_num_threads"),
        ("paddle", "EngineConfig.paddle.cpu_math_library_num_threads"),
    ],
)
def test_rapidocr_num_threads_propagated_per_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    engine_key: str,
):
    params = _capture_params(monkeypatch, RapidOcrOptions(backend=backend), tmp_path)
    # num_threads must reach the engine actually in use, not only ONNXRuntime.
    assert params[engine_key] == 4


@pytest.mark.parametrize("backend", ["onnxruntime", "paddle", "torch"])
def test_rapidocr_gpu_device_uses_cuda_ep_cfg_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(backend=backend),
        tmp_path,
        resolved_device="cuda:2",
    )
    # The GPU device id must use the engine's real key; the legacy top-level
    # `gpu_id` key is not read by RapidOCR (see #3049 for the torch fix).
    assert f"EngineConfig.{backend}.cuda_ep_cfg.device_id" in params
    assert params[f"EngineConfig.{backend}.cuda_ep_cfg.device_id"] == 2
    assert params[f"EngineConfig.{backend}.use_cuda"] is True
    assert f"EngineConfig.{backend}.gpu_id" not in params


def test_rapidocr_pins_explicit_model_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch, RapidOcrOptions(backend="onnxruntime"), tmp_path
    )
    # Paths are always pinned now, so rapidocr never lazy-resolves models.
    assert params["Det.model_path"] is not None
    assert params["Rec.model_path"] is not None
    assert "Det.lang_type" not in params
    assert "Rec.lang_type" not in params


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:2"])
def test_rapidocr_engine_accelerators_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, device: str
):
    params = _capture_params(
        monkeypatch, RapidOcrOptions(), tmp_path, resolved_device=device
    )
    assert params["EngineConfig.onnxruntime.use_dml"] is False
    assert params["EngineConfig.onnxruntime.use_coreml"] is False
    assert params["EngineConfig.torch.use_mps"] is (device == "mps")
    assert "Det.use_dml" not in params


def test_rapidocr_directml_auto_uses_actual_ort_capability(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import onnxruntime

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        "docling.models.stages.ocr.rapid_ocr_model.platform.system", lambda: "Windows"
    )
    params = _capture_params(
        monkeypatch, RapidOcrOptions(), tmp_path, requested_device="auto"
    )
    assert params["EngineConfig.onnxruntime.use_dml"] is True


def test_rapidocr_explicit_coreml_override_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(rapidocr_params={"EngineConfig.onnxruntime.use_coreml": True}),
        tmp_path,
    )
    assert params["EngineConfig.onnxruntime.use_coreml"] is True


def test_rapidocr_missing_explicit_cuda_fails_before_loading_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with pytest.raises(RuntimeError, match="no CUDAExecutionProvider"):
        _capture_params(
            monkeypatch,
            RapidOcrOptions(),
            tmp_path,
            requested_device="cuda:2",
            resolved_device="cuda:2",
            available_providers=["CPUExecutionProvider"],
        )


def test_rapidocr_auto_cuda_does_not_require_cuda_torch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(),
        tmp_path,
        requested_device="auto",
        resolved_device="cpu",
        available_providers=[
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    assert params["EngineConfig.onnxruntime.use_cuda"] is True
    assert params["EngineConfig.onnxruntime.use_dml"] is False


def test_rapidocr_explicit_cpu_override_does_not_require_cuda_ort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    params = _capture_params(
        monkeypatch,
        RapidOcrOptions(rapidocr_params={"EngineConfig.onnxruntime.use_cuda": False}),
        tmp_path,
        requested_device="cuda:2",
        resolved_device="cuda:2",
        available_providers=["CPUExecutionProvider"],
    )
    assert params["EngineConfig.onnxruntime.use_cuda"] is False
