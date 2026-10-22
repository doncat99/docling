# Upstream performance audit: fork applicability

Source verification and local diagnostics on 2026-10-22. These results describe source checkouts
and isolated diagnostics, not acceptance of a rebuilt OntoScience installation.

| Upstream item | Current fork status | Remaining boundary |
| --- | --- | --- |
| docling-ibm-models #161: host-controlled tag decoding | Partly addressed by verified block decoding, direct SDPA and KV caching | Each verified block still materializes predictions on the host; candidate history and acceptance remain host controlled |
| docling-ibm-models PR #193: direct SDPA | Equivalent optimization already present | Do not count it as a new gain |
| docling-ibm-models PR #194: BBox repeated work | Image/language projections and gates already hoisted; chunked computation present | Further changes require output and performance comparison |
| docling-ibm-models PR #198: unused V2 BBox computation | Our exercised path is V1 | Not a demonstrated defect of the exercised model path |
| docling #4490: repeated PDF tiling-pattern expansion | Present in consumed docling-parse 7.22.3 | Not triggered by the 77-page report; see below |
| docling #4162: global bookmark matching | Destination-page loss already fixed upstream by #4244; page index now avoids irrelevant candidate visits | Genuinely pageless bookmarks still require global fuzzy matching |

## Bookmark matching

See [heading levels](heading_levels.md#matching-cost-and-page-index-regression-coverage).
The landed index preserves unknown-page candidates and original tie order. It changes neither the
matching threshold nor the enabled-by-default policy. Focused outline and heading suites report
67 passed and 2 skipped. The skips are not passing evidence.

## Tiling patterns (#4490)

The v7.22.3 `pdf_decoder<STREAM>::do_pattern_fill` still reparses a tiling cell and replays it
over the covered lattice. The shape-skip guard applies to shading patterns, not this tiling path.

Decoded xref dictionaries of the 77-page Vermilion report contain no `/Type /Pattern` or
`/PatternType 1` objects, including compressed objects. This issue therefore does not explain that
report's current latency.

A single-worker synthetic PDF with a diagonal-line tiling cell and repeated identical fills
confirms the cost exists in the consumed native parser. With rendering, 10 fills took 0.071 s
and 100 took 0.600 s; without rendering they took 0.008 s and 0.071 s. These are exploratory
single observations, not a stable speed benchmark or an upstream-versus-candidate comparison.
Parser initialization is included; the first observation is visibly affected by initialization.

Upstream docling-parse PRs #386 and #390 were open at inspection time. They cannot be adopted
as output-equivalent fixes without checking pattern text, bitmap placement, extracted shape
lines and rendered geometry. In particular, skipping pattern replay or replacing expanded
strokes with a render pattern changes extraction semantics. No parser patch was applied here.

## Host-controlled decoder (#161)

The fork has already reduced per-label host feedback to per-verified-block feedback. Moving
acceptance checks to a tensor alone cannot eliminate feedback while Python owns the next draft
and cache cursor. A device-resident implementation would need to carry those dependencies too.

Existing isolated replays do not establish a deployable improvement:

- Deferring reads until table completion uses predetermined future inputs, so it is a
  counterfactual replay rather than an executable decoder. Six alternating observations did not
  show a consistent timing advantage.
- Adaptive draft width reduced computed positions from 28,384 to 13,340 but increased decoder
  calls from 887 to 1,224. Three decode-only observations were about 9.0–9.4 s versus baseline
  5.5–6.4 s. Less arithmetic alone did not reduce this serial path.
- A fused residual/normalization experiment preserved observed tags but changed hidden-state
  values and did not establish stable full-report acceleration. It remains experimental.

Host materialization spans include waiting for previously queued GPU computation. They must not
be reported as pure transfer overhead. None of these replays demonstrates a new end-to-end gain,
and no new decoder implementation was adopted in this audit.

Local reproducibility evidence is retained under the OntoScience workspace `.tmp/`:
`heading-4162-cfa2025/landed-page-index.json`, `pattern-4490-check.py`,
`pattern-4490-check.json`, `decoder-control-replay-oct22.json`,
`decoder-final-candidates-oct22.json` and `decoder-input-replay-oct22.json`.
These ignored diagnostic files are evidence, not dependency or runtime inputs.
