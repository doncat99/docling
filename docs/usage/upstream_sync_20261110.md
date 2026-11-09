# Upstream synchronization: 2026-11-10

## Revisions

Merge upstream `1005c8d7d39447ff433ff650fbc49e046ad0aeee` (162 incoming
commits) into fork `f7d9f5136f887d03d6bdf43b66f33e60625fe567`.
Preserve all 14 fork commits. The distribution version becomes
`2.149.0+ontoscience.1`.

Incoming updates include PDF script positions and text normalization, OCR
orientation retries and broken text replacement, degenerate table handling,
page resource release, reading order, Office parsing, archive limits, and
CLI format and model presets.

## Conflict decisions

Resolve all six conflicted files in the current owners:

- Object detection: keep upstream dtype transfer and fork accelerator
  serialization; move inputs inside the accelerator guard.
- RapidOCR: keep upstream orientation retries; guard the actual reader call.
- PDF pipeline: keep upstream page release, but retain parsed pages until
  native formula enrichment finishes when that fork feature is enabled.
- Heading documentation: retain parser outlines and the fork page index.
- Package metadata: retain Pillow below 14 and the immutable fork model wheel
  `4.0.4+ontoscience.3`; require upstream `docling-core>=2.106,<3`.
- Lock file: regenerate from the merged metadata and verify lock consistency.

Keep MPS serialization, RT-DETR folding, native formulas, and the pinned model
wheel. Do not substitute upstream model weights or discard fork changes.

## Validation and limits

- `make validate`: passed, including Ruff, ty, Tach, module coverage, dprint,
  and lock consistency. The existing environment reports type warnings; this
  is not a claim that all optional dependencies have been rebuilt.
- Focused integration tests with `docling-core==2.106.0`: 224 passed,
  2 skipped. Covers MPS concurrency, OCR guard, parsed-page release, headings,
  native formulas, script positions, degenerate tables, input documents, and
  multiline OCR. The old-core footnote failure passes with the required core.
- `uv lock --locked --offline`: passed.
- Conflict resolution changes pass `git diff --check` against upstream.
  Upstream reference fixtures retain their original whitespace.

The complete upstream test matrix and full-document model performance were
not run. This sync changes the fork source only; it does not update the
OntoScience artifact pin or the installed App. Case delta: no-case, dependency
source synchronization without a new product contract; focused integration
tests provide supporting evidence.
