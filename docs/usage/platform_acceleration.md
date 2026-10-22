# Platform acceleration: capabilities, selection, and evidence

An installed accelerator library does not prove that a model used it. Distinguish
the dependency build, runtime availability, requested configuration, actual session
providers, executed operators, and measured workload quality/latency.

## Owners and current policy

| Stage | Engine | macOS Apple Silicon | Linux / Windows NVIDIA | Windows other GPUs |
| --- | --- | --- | --- | --- |
| Layout and TableFormer | PyTorch | MPS when available; shared MPS submission guard | CUDA only with CUDA Torch and a working driver | CPU; ONNX DirectML does not accelerate Torch models |
| OCR detection, classification, recognition | RapidOCR / ONNX Runtime | CPU by default; Core ML opt-in after workload measurement | CUDA EP preferred in auto mode when installed | DirectML EP in auto mode when installed; requires the RapidOCR session fix below |
| OCR with explicit Torch backend | RapidOCR / PyTorch | Forward `use_mps`; retain the shared MPS guard | Forward CUDA settings | CPU unless a separately supported backend is supplied |
| PDF extraction, rendering, matching | native CPU libraries | CPU | CPU | CPU |

Core ML is Apple's inference framework, available on Apple platforms including
macOS and iOS. Its execution provider can delegate work to CPU, GPU and Neural
Engine according to model support and provider options. It is distinct from
PyTorch's MPS backend. A Core ML session may still execute unsupported nodes on
CPU. Selecting Core ML does not promise exclusive GPU execution or lower latency.

RapidOCR configuration belongs to `RapidOcrModel`; inference/session behavior
belongs to RapidOCR. Native packaging belongs to Semantica. Do not duplicate
device selection in application callers or import an external source checkout
as a packaged runtime dependency.

## Configuration repairs

- ONNX auto selection consults ONNX Runtime providers independently of the Torch
  build. A CPU Torch package must not hide an available OCR CUDA provider.
- DirectML is configured through `EngineConfig.onnxruntime.use_dml`; the old
  `Det.use_dml`, `Cls.use_dml` and `Rec.use_dml` keys were ignored.
- Explicit CUDA with no CUDA EP fails with a dependency error rather than silently
  promising GPU OCR. Per-engine user overrides are applied before this check.
- Torch OCR forwards `EngineConfig.torch.use_mps`; constructor and inference use
  the same shared MPS guard as layout and table inference.
- Explicit `rapidocr_params` remain authoritative. Core ML is deliberately not
  auto-enabled on the strength of an installed provider alone.

RapidOCR's ONNX session must set `enable_mem_pattern=False` and
`execution_mode=ORT_SEQUENTIAL` for DirectML. The upstream revision inspected on
2026-10-22 lacked these settings; the correction lives in doncat99/RapidOCR.
Do not release automatic DirectML selection without that corrected dependency.
See the [ONNX DirectML requirements](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html).

## macOS measurements, 2026-10-22

ONNX Runtime 1.23.2, RapidOCR 3.9.2, four intra-op threads, identical pinned
PP-OCRv6 small detection/recognition and v2 classification models. A rotated
dense-text fixture produced the same three recognized lines on CPU and Core ML.
After the first invocation, the medians were 0.894 s on CPU and 3.549 s on Core ML.
This is a small diagnostic fixture, not a full-report performance claim.

An independent profiling run confirmed actual Core ML and CPU operator events
in all three sessions. Core ML events for detection/classification/recognition
were 44/44/56, alongside 76/44/84 CPU events across four calls. These are fused
execution events, not percentages of original model operations. Available EPs
or `session.get_providers()` alone cannot establish this distinction. Compilation
caches were already present; these measurements do not establish cold-start cost.

Additional fixtures rendered at 216 dpi with PyMuPDF, four invocations per engine
with profiling enabled (last three median):

| Fixture | CPU | Core ML | Comparison |
| --- | --- | --- | --- |
| Vermilion ESG report page 16 | 4.858 s | 18.523 s | 100 lines, identical text and boxes |
| Chinese/English/numeric synthetic page | 0.405 s | 0.722 s | 3 lines, identical text and boxes |

These diagnostic measurements are sequential, not an isolated randomized benchmark.
They establish functioning mixed execution and no observed output difference on
these inputs; they do not establish a universal CPU advantage or OCR accuracy on
all documents. Core ML remains an explicit workload-specific option.

Diagnostic evidence is local under `.artifacts/ocr-platform/`. Do not check in
private report pages, extracted report text, or absolute local input paths.

## Deployment and acceptance boundaries

Current Semantica Linux/Windows artifacts contain CPU Torch and CPU ONNX Runtime.
Changing selectors cannot make those artifacts GPU-capable. macOS MPS does not
imply ONNX OCR uses the GPU. Semantica's hosted native acceptance intentionally
uses CPU; its receipt must label CPU validation separately from capability inventory.

CUDA needs matching Torch, torchvision, ONNX GPU runtime, CUDA/cuDNN libraries
and driver compatibility. DirectML requires a Windows DirectML runtime and its
session constraints. Never install multiple ONNX Runtime distributions in the
same Python prefix: they overwrite the same import namespace. Deliver separately
pinned native variants, not mutable installation into an active environment.

Before promotion, each GPU artifact needs real matching hardware acceptance:
offline model loading, actual session providers, operator profiling where needed,
multilingual OCR quality, boxes/numbers, cold and warm latency, memory, restart
and failure behavior. An unavailable driver or CPU fallback is not a GPU pass.
CUDA and DirectML hardware acceptance has not been performed in this session.
The original 77-page ESG parse did not use OCR; OCR timings do not change its
previously measured layout/TableFormer performance.
