# MPS diagnostic troubleshooting

For controlled optimization measurements and output comparisons, see
[MPS performance evidence](mps_performance.md). The submission guard below
remains required after those optimizations.

## Metal assertion during concurrent PDF profiling

Investigation date: 2026-10-22.

```text
Scheduled handler provided after commit call
```

This native Metal assertion aborts the process. A Python exception handler cannot
recover the conversion.

The observed trigger was a diagnostic wrapper that called
`torch.mps.synchronize()` before and after TableFormer prediction while the
threaded pipeline could submit layout work to MPS. The wrapper enabled these
calls by default. They were instrumentation, not part of the normal conversion
path. Both PyTorch 2.11.0 and 2.14.1 aborted with this instrumentation.

Removing these injected barriers allowed one full document run to complete.
That success did not establish a fix. A fresh run without any injected barrier
aborted with a second assertion:

```text
A command encoder is already encoding to this command buffer
```

A minimal two-thread PyTorch program reproduced the second assertion without
Docling, PDF input, model weights, or explicit synchronization. One thread ran
Conv2d/BatchNorm; the other ran `arange` and tensor arithmetic. The native crash
stack entered `MPSStream::executeMPSGraph` from `batch_norm_mps_out`; the other
thread submitted `arange_range_fill_mps`. Separate native MPS streams also failed.

This locates the reproducible defect in concurrent PyTorch/Metal command
encoding on this configuration. It does not identify the exact faulty source
line in PyTorch or the Apple driver. The two assertion messages can reflect
different invalid command-buffer states; their precise internal relationship
is not established. Explicit synchronization is not a necessary trigger.

## Correction

- Remove explicit device-wide synchronization from concurrent model callbacks.
  The local benchmark no longer has an option to enable this path.
- Keep both model stages on MPS. Use the same process-wide
  `accelerator_inference_context` in the Transformers object-detection engine
  and TableFormer. It serializes host inference submissions, including input
  transfers and CPU materialization of outputs, across model instances.
  CPU/CUDA calls do not acquire this lock. Pipeline queues, PDF extraction, and
  table preparation remain independent of the lock.
- Inject the same context factory into `TFPredictor(inference_context=...)`.
  Protect model loading, device transfers, prediction, tensor postprocessing,
  and output materialization. Run CPU image preparation, HTML conversion, and
  cell matching outside the guard. Layout also prepares CPU inputs before
  acquiring the guard and builds output objects after materialization.
  The standalone predictor defaults to `nullcontext`; callers that share MPS
  across threads must supply their shared submission guard.
- This is a backend workaround with a concurrency cost. It does not repair
  PyTorch or turn either model into a CPU model. It cannot protect unrelated
  third-party MPS calls that bypass the context. Other MPS engines need their
  own evidence and integration before claiming app-wide protection.
- Measure conversion wall time around converter initialization and `convert()`.
  Per-stage host timings can overlap and are not isolated GPU kernel timings.
  Do not sum them to infer document latency.
- For isolated kernel timings, replay captured inputs in a separate process with
  no concurrent model submission. Synchronization in that isolated replay is a
  different experiment from adding barriers to the threaded pipeline.

The fix includes removal of unsafe diagnostic instrumentation and protection
of the two affected production inference paths. Existing device-selection and
TableFormer algorithm changes are separate work. Remove the guard only after
the native regression and repeated full-document runs pass on a backend with
a demonstrated fix.

## Evidence

Hardware: Apple M2, 24 GB RAM, macOS 15.6.1, arm64; four CPU threads.
Document: 77-page Vermilion report; SHA-256:

```text
12655a44a303dbd7cb24e16ccd204afbda6624ddf45bd3234d40f0358a5c3410
```

OCR was disabled. TableFormer used accurate weights. CPU fallback was not enabled.

| Configuration | Result |
| --- | --- |
| PyTorch 2.11.0, both stages MPS, injected synchronization | Metal abort |
| PyTorch 2.14.1, both stages MPS, injected synchronization | Metal abort |
| PyTorch 2.14.1, both stages MPS, no injected synchronization | SUCCESS, 77 pages, 37 tables, 174.290 seconds |
| Fresh fork run, PyTorch 2.14.1, both stages MPS, no injected synchronization, no guard | Command-encoder assertion, exit 134 |
| Two-thread Conv2d/BatchNorm and arange, PyTorch 2.14.1, no guard | Same command-encoder assertion, exit 134 |
| Same two-thread workload with distinct native MPS streams | Same command-encoder assertion, exit 134 |
| Fork with shared MPS submission guard, fresh process 1 | SUCCESS, 77 pages, 37 tables, 5877 cells, zero errors, 41.077 seconds |
| Fork with shared MPS submission guard, fresh process 2 | SUCCESS, 77 pages, 37 tables, 5877 cells, zero errors, 41.783 seconds |

Both guarded runs used PyTorch 2.14.1 and the working fork sources, with layout
and TableFormer on MPS. Their complete exported documents match the prior
complete reference after removing only bounding-box fields, including all cell
text and structure. Run 1's maximum bounding-box difference was 0.017214 PDF
points. This is reference equivalence, not independently verified ground truth:
the existing warning about one unmatched PDF cell in a 46-by-9 grid remains.
These times are validation observations, not a controlled speedup comparison.

Focused validation: nine tests passed across
`test_mps_concurrent_inference.py` and `test_table_structure_device.py`; scoped
Ruff, formatting, and diff checks passed. Full `make validate` was blocked
before running hooks by the existing `models-local` direct-reference metadata
error requiring `tool.hatch.metadata.allow-direct-references=true`. This gate
is not reported as passed. No installed application was rebuilt in this check.

The successful run above used Docling 2.139.0 and docling-ibm-models 4.0.3.
Later optimized-model runs also completed without injected synchronization.
Do not compare their latency to this row as an algorithm speedup unless the
model code, options, hardware load, and measurement boundaries match.

Original local evidence names (not distributed with this fork):

- `vermilion-device-mps-torch211-crash.log`
- `vermilion-device-mps-torch214-concurrent-crash.log`
- `vermilion-device-mps-concurrent-nosync/metadata.json`
- `vermilion-device-mps-concurrent-nosync/summary.json`
- `vermilion-device-mps-concurrent-nosync/document.json`
- `vermilion-mps-sync-fix-validation.log` and its `metadata.json`
- macOS crash report incident `8A82B889-94A0-4094-A5D3-F5FE85FCF5DB`
- `vermilion-mps-submission-guard-validation/{metadata,summary,document}.json`
- `vermilion-mps-submission-guard-validation-repeat/{metadata,summary,document}.json`

The native regression test is
`tests/test_mps_concurrent_inference.py`. It contains the minimal workload with
the guard and checks numerical outputs in a child process. An abort fails the
test without killing pytest. Its CPU/CUDA context tests also check that the
workaround does not serialize those devices (no GPU required for those tests).

## Repeat the corrected diagnostic

Use the fork and its compatible model dependency in an isolated environment.
Do not change an installed application's Python packages for this test.
Run from the fork root, with local model artifacts and the PDF above:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python docs/examples/mps_pipeline_diagnostic.py \
  /path/to/report.pdf /path/to/new-evidence-directory \
  --artifacts-path /path/to/models \
  > /path/to/run.log 2>&1
```

The script rejects unavailable MPS, explicit CPU fallback, and reused output
directories. It records versions, input hash, complete options, conversion
status, counts, errors, timings, and the exported document. Keep the log to
confirm both model device selections. A completed process alone is insufficient:
check SUCCESS, zero errors, all 77 pages, 37 tables, and table content against a
reference. Matching counts alone does not establish output equivalence.

Repeat in fresh processes. Record each result, including native aborts and
memory failures. A successful run is evidence for that run, not a guarantee
that all MPS workloads are stable. If the assertion returns without injected
barriers, capture the native crash report, exact dependency revisions, options,
input hash, and model device logs before changing pipeline concurrency.
