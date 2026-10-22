# Upstream synchronization: 2026-10-22

## Revisions and integration decisions

Merge upstream `10054ec7bbdc9435ef7ca2ec6be90c782082ae52` (40 incoming commits)
into fork `fd3269da328d9149ff28d45a0525487a17eaa93c`.
The fork distribution version becomes `2.141.0+ontoscience.1`.

Incoming changes include PDF form-field extraction, bounded reading-order
processing and document deadlines, OTSL cell preservation, Office/LaTeX parsing
fixes, OCR copy reductions, and service download validation fixes.

The PDF pipeline keeps both the fork's native formula model and upstream's
optional form-field stage. It also keeps upstream's document-deadline propagation.
The universal lock is regenerated for the merged package metadata.

The shared MPS guard, table device selection, layout Conv/BatchNorm folding,
and native formula implementation remain unchanged from the fork base.
The model wheel remains hash-pinned to `4.0.4+ontoscience.2`; upstream PyPI
models are not substituted. This preserves KV caching, vectorized BBox inference,
verified speculative decoding, position-aware candidates, spatial matching,
cell-ID indexing, and the extended tag budget.

## Full-report comparison

Use the 77-page Vermilion report with SHA-256
`12655a44a303dbd7cb24e16ccd204afbda6624ddf45bd3234d40f0358a5c3410`.
Run three alternating baseline/candidate pairs in fresh processes. Both sides
use PyTorch 2.14.1, four CPU threads, MPS for layout and accurate TableFormer,
the same weights, no OCR, and native formula processing. Concurrency settings
are identical. Conversion time includes model initialization, excludes JSON
export, and includes the same diagnostic instrumentation on both sides.

| Pair | Fork base | Synced fork |
| --- | ---: | ---: |
| 1 | 27.984 s | 28.690 s |
| 2 | 28.527 s | 28.609 s |
| 3 | 28.519 s | 28.634 s |
| Median | 28.519 s | 28.634 s |

The observed median difference is +0.41%; this small sample establishes neither
a speedup nor statistical equivalence. All six runs complete with 77 pages,
37 tables, 5877 cells, zero errors, and no MPS crash. Complete exported content
matches after excluding only bbox fields and the root schema version. All bbox
paths and values match exactly. Raw tag-sequence multisets match, including
the long table beyond the old 1024-tag limit (maximum raw length 1083, including
the diagnostic sequence markers).

Sampled MPS driver memory peaks at 2,621,177,856 bytes in every run (100 ms
sampling; not a guaranteed hardware peak). Process maximum RSS ranges from
1.875 to 2.490 GB for baseline and 1.968 to 2.149 GB for candidate. These
measurements do not establish a memory reduction.

Baseline installs `docling-slim==2.139.0+ontoscience.4` and
`docling-core==2.100.0`; candidate installs `2.141.0+ontoscience.1` and
`2.102.0`. Each uses the same pinned models wheel. Runtime module paths are
recorded to verify that both processes load installed wheels, not the checkout.

## Checks and evidence

- `make validate`: passed, including Ruff, ty, Tach, coverage, dprint and lock.
- `uv lock --locked`: passed.
- Focused tests: 86 passed, 1 skipped. Scope: MPS concurrency, table device,
  native formulas, OTSL, form-field ordering, reading order, layout overlap,
  picture RGB conversion, and Office image modes.
- Full-report evidence remains local under
  `.artifacts/upstream-sync-20261022/`: `run_series.py`, `run_report.py`,
  `results.json`, six report directories and logs. Validation logs are
  `/tmp/docling-sync-{validate,tests}-20261022.log`.

This is fork integration validation. The complete upstream test matrix and
CPU/CUDA performance comparisons were not run. No App dependency pin, installed
application, acceptance baseline, or published release asset is changed here.
