# MPS inference optimization evidence

Investigation: 2026-10-22. These are source-fork measurements, not acceptance of
an installed OntoScience application. Keep the shared MPS submission guard
described in [MPS diagnostics](mps_diagnostics.md).

## Remaining work and selected changes

An isolated replay of all 37 real table images took 12.805 seconds: image
encoding 0.430 seconds, tag encoding 0.811 seconds, and cell bounding-box
prediction 3.059 seconds. The remaining time includes structure decoding,
merging, and dispatch. The fixed-width-16 draft decoder made 1299 verification
calls for 9180 output tags including EOS, proposing 20784 positions. Acceptance
fraction alone is not the optimization objective: width 8 made more calls and
was slower. Captured-memory replay at widths 8, 16, 32, and 64 preserved all
37 tag sequences. Width 32 reduced calls to 1058; width 64 reduced calls further
but was slower than 32 in the follow-up sweep. These are workload observations,
not a universal optimal-width claim.

The actual Heron layout checkpoint uses an RT-DETRv2 ResNet backbone. Isolated
same-input replay located roughly 60% of model time in the backbone. Folding
55 eval-only convolution/normalization pairs reduced complete layout inference
from 146.7–157.6 ms to 104.6–108.1 ms over four alternating rounds, eight calls
per round after warmup. Logit and normalized-box maximum differences were
0.001209 and 0.00008535. These are floating-point transformations, not bitwise
equivalence. They require downstream document comparison.

The selected changes are:

- Fold only the explicit Transformers RT-DETR ResNet convolution and shortcut
  types in MPS inference. Use PyTorch's convolution/BatchNorm folding utility.
  Frozen BatchNorm uses its implementation's fixed epsilon of `1e-5`.
  Load the checkpoint and enter eval mode first. The folded runtime is not
  intended for training or checkpoint export. Original checkpoint files,
  inference precision, model architecture, and accuracy mode stay the same.
- Use verification width 32 on MPS. Every accepted tag must still pass the
  causal greedy-model verification, existing tag corrections, and EOS checks.
  CPU and CUDA keep width 16 pending measurements on those backends.

No OCR, reduced precision, different model weights, CPU fallback, cross-table
batching, or unguarded concurrent MPS submission was used. A separate BBox
matrix-product experiment was not selected. The later lock-scope decision is
recorded below; its full-document speedup is not established.

## Three alternating full-report pairs

Apple M2, 24 GB RAM, macOS 15.6.1, PyTorch 2.14.1, four CPU threads, float32.
Both layout and accurate TableFormer use MPS. Each conversion uses a fresh
process, the same input, options, weights, instrumentation, and guard. No GPU
benchmark processes overlap. Wall time includes converter/model initialization
and excludes JSON export. Both sides already contain the earlier KV caching,
vectorized BBox, and verified draft algorithm.

Input: 77-page Vermilion report, SHA-256
`12655a44a303dbd7cb24e16ccd204afbda6624ddf45bd3234d40f0358a5c3410`.

| Round | Unfolded layout, width 16 | Folded layout, width 32 |
| --- | ---: | ---: |
| 1 | 37.116 s | 33.056 s |
| 2 | 39.605 s | 34.669 s |
| 3 | 39.129 s | 32.648 s |
| Median | 39.129 s | 33.056 s |

The observed median reduction is 15.5%. All six conversions succeeded with
77 pages, 37 tables, 5877 cells, and zero conversion errors. Complete exported
documents match after removing only `bbox` fields. Raw tag-sequence multisets
also match. Maximum coordinate difference is 0.000145 PDF points; all coordinate
paths match. The existing warning about one unmatched PDF cell remains.
Reference equality is not human ground truth, and this is not a broad speed
guarantee or a comparison against the original upstream decoder.

## Other document regression checks

Three existing PDF fixtures use the same full-conversion options. Each row is
one fresh-process baseline/candidate pair, not a repeated latency distribution.

| Fixture | Pages | Tables / cells | Baseline | Candidate | Maximum bbox difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2206.01062.pdf` | 9 | 5 / 420 | 8.648 s | 4.772 s | 0.0000611 PDF points |
| `redp5110_sampled.pdf` | 18 | 6 / 198 | 10.545 s | 6.004 s | 0.0001221 PDF points |
| `normal_4pages.pdf` | 4 | 1 / 15 | 3.739 s | 2.054 s | 0.0000611 PDF points |

All six conversions succeeded with zero errors. Each candidate's complete
export matches its baseline after removing only `bbox`; raw tag multisets and
all bbox paths also match. These cover different paper/technical-document
layouts but do not establish correctness for all reports or human ground truth.
Evidence: `multidoc-oct22-{fixture}-{baseline,candidate}/` and sibling logs.

The integrated fork then ran the normal `mps_pipeline_diagnostic.py` entrypoint
without experimental monkeypatches: 35.524 seconds, 77 pages, 37 tables,
5877 cells, zero errors. Its complete document, including bbox fields, exactly
matched candidate round 1. This is an integration check, not a fourth paired
performance observation.

## Memory and validation boundaries

Focused tests passed: 26 model-fork tests and 18 Docling tests, including
CPU/MPS convolution-folding equality, rejection/rollback at widths 16/32/64,
native MPS concurrency, and device/budget propagation. Model-fork
`make validate` passed. Docling scoped Ruff/format and module-coverage checks
passed. Full Docling `make validate` remains blocked before hooks by the
existing `models-local` direct-reference metadata error. Its local environment
also lacks `tach`; that gate is not passed. These are separate from the completed
conversion checks. No installed application or product acceptance baseline was
changed or promoted.

macOS `/usr/bin/time -l` measured peak process footprints of 4.798–4.818 GB
for baseline and 4.743–4.822 GB for candidate. Maximum resident set sizes were
2.037–2.145 GB and 1.864–2.330 GB respectively. There is no demonstrated memory
reduction. Driver-memory samples taken after table calls were at most 2.619 GB
on both sides; these samples are not GPU peaks.

Local evidence (not distributed with this fork):

- `docling-inference-replay-oct22/{inputs.pt,table-replay.json,layout-replay.json}`.
- `layout-fold-resnet-experiment.log` and `decode-width-*-oct22*.jsonl`.
- `optimization-oct22-{baseline,candidate}-{1,2,3}/` with complete documents,
  metadata, summaries, and raw table predictions; sibling logs contain process
  memory measurements.
- `optimization-oct22-comparison.json` records full-field comparisons.

Do not add device-wide synchronization inside the threaded conversion to obtain
these timings. The isolated replay uses barriers only after conversion workers
have ended. Keep full-report timings and isolated component timings distinct.

## Narrower submission guard: retained, no stable speedup claim

Six alternating pairs used the same 77-page report, both models on MPS,
PyTorch 2.14.1, float32, and four CPU threads. GPU benchmarks did not overlap.
Other system activity remained; these are not isolated performance baselines.

| Pair | Coarse guard | Narrow guard |
| --- | ---: | ---: |
| 1 | 29.88 s | 28.63 s |
| 2 | 29.71 s | 29.05 s |
| 3 | 29.62 s | 29.27 s |
| 4 | 29.77 s | 30.19 s |
| 5, complete lock timing | 32.34 s | 34.10 s |
| 6, complete lock timing | 29.29 s | 30.28 s |

The first four pairs reduced median wall time by 1.9%; the last two were
slower. Across all six pairs the medians were effectively equal. Summed
layout/table host lock waits in pairs 5 and 6 fell from 12.90 to 9.34 seconds
and 11.68 to 9.47 seconds. CPU preparation and matching overlapped layout's
held-guard intervals. These host intervals are not GPU execution timings and
must not be added to infer critical-path savings.

All twelve conversions produced 77 pages, 37 tables, 5877 cells, and zero
conversion errors or crashes. Complete exports matched after removing only
bbox fields; raw tag-sequence multisets matched (9180 tags including EOS).
Maximum bbox deviation was 0.0000611 PDF points. The existing unmatched-cell
warning remains. Reference equality is not independently verified ground truth.

Retain the narrower guard to eliminate unnecessary CPU serialization. Keep all
device transfers, inference, tensor postprocessing, and materialization under
the shared guard. This does not remove MPS crash protection, increase worker
counts, or establish a stable end-to-end speedup. Regression tests check both
predictor entry points and the layout boundary. Local evidence:
`mps-lock-retest-oct22-results.json` in OntoScience's ignored `.tmp/` directory.

The integrated source revision completed a fresh full-report run in 36.457
seconds: 77 pages, 37 tables, 5877 cells, zero errors. Content and raw tags
matched the coarse-guard reference; maximum bbox delta was 0.0000611 PDF
points. This is an integration observation, not a paired speedup measurement.
Focused validation passed 21 Docling tests and 53 model-fork tests, including
the native MPS guard regression. The required model artifact is
`docling-ibm-models==4.0.4+ontoscience.2`; older artifacts do not accept the
injected inference context. No installed application was rebuilt.

For this revision, changed-file `make validate` passed, including Ruff, ty,
Tach, module coverage, maximum-line checks, and lock validation. The earlier
metadata blocker was corrected by allowing the fork's exact direct artifact
reference. The Docling wheel also built successfully. Earlier blocked-gate
reports above describe their original revisions and are not current results.
