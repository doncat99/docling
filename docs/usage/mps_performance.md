# MPS inference optimization evidence

Investigation: 2026-10-22. These are source-fork measurements, not acceptance of
an installed OntoScience application. Keep the shared MPS submission guard
described in [MPS diagnostics](mps_diagnostics.md).

## Follow-up: redundant decoder masks and incomplete layout folding

The model fork now creates one additive causal mask per verified block, instead
of asking SDPA to convert the same boolean mask in each of six layers. Two
alternating fresh-process pairs, including model initialization, measured
30.434/31.760 seconds for boolean masks and 28.621/28.807 seconds for shared
additive masks. The observed median reduction is 7.7%. All four complete
documents, including bbox, and raw table-tag multisets were exactly equal.
See the model fork's `docs/tableformer-speculative-decoding.md` for boundaries.

Layout fusion previously covered only 55 ResNet normalization pairs. The Heron
checkpoint also has 42 fixed BatchNorm operations in its encoder, projections,
and RepVGG branches. Twelve RepVGG blocks evaluate two linear convolutions
before their shared activation. Fold their normalization and combine the padded
1x1 kernel with the 3x3 kernel, including both biases. This removes twelve
convolutions and twelve branch additions. Preserve the final activation;
branches with an internal nonlinearity must remain separate. Restrict rewriting
to explicit RT-DETR types and their owned projections. Other models remain
untouched. The transformation is idempotent and inference-only.

Fold fixed weights on CPU after checkpoint loading and eval, before the single
guarded device transfer. Layout inference remains on MPS. An isolated four-round
comparison of the same 109 transformations and transfer measured:

| Round | Transfer then GPU folding | CPU folding then transfer |
| --- | ---: | ---: |
| 1 | 0.2859 s | 0.1243 s |
| 2, CPU first | 0.1831 s | 0.1161 s |
| 3 | 0.2344 s | 0.1887 s |
| 4, CPU first | 0.1689 s | 0.1799 s |

This initialization saving is small. It does not explain multi-second wall-time
variation or establish an end-to-end speedup. Isolated layout replay with GPU
folding also had mixed small timing changes. Maximum logit and normalized-box
differences were 0.0005622 and 0.00009467; convolution reassociation changes
floating-point rounding and requires downstream quality checks.

Three fresh-process expanded-GPU-fold conversions completed the 77-page report
in 46.121, 34.942, and 28.369 seconds; the nearby ResNet-only controls took
31.849, 27.970, and 28.931 seconds. The first control overlapped a short GPU
regression test and is excluded from performance conclusions. Other system
activity included a virtual machine and metadata indexing. Do not select the
fastest observation or describe this set as a stable speedup. All six exports
preserved every field except 72 bbox numbers, with maximum difference
0.0005951 PDF points. Raw tag multisets matched; all runs returned 77 pages,
37 tables, 5877 cells, and zero conversion errors. The existing unmatched-cell
warning remains.

Additional full baseline/GPU-fold pairs preserved all non-bbox fields and raw
tags: `normal_4pages` (4 pages, 1 table, 15 cells), `2206.01062` (9 pages,
5 tables, 420 cells), and `redp5110_sampled` (18 pages, 6 tables, 198 cells).
Maximum bbox differences were 0.0000611, 0.0000611, and 0.0001221 PDF points.
Candidate wall times were slower for all three; these are quality checks,
not demonstrated performance wins.

The final CPU-fold-before-transfer path then completed two fresh 77-page
conversions in 30.748 and 29.321 seconds. Its nearby old-initializer control
took 26.992 seconds. Both candidates preserved all non-bbox fields, raw tag
multisets, counts, and zero errors; maximum bbox drift remained 0.0005951 PDF
points. Rechecking all three fixtures through this final path also preserved
all non-bbox fields and raw tags, with the same maximum bbox deltas above.
Their wall times were 4.865, 6.517, and 7.630 seconds versus 4.153, 4.363,
and 5.220 seconds for the controls. These results do not prove that removing
redundant layout operators improves full-document latency. Do not combine the
mask-only 7.7% observation with an assumed layout gain. Further controlled
profiling is required before making an overall speed claim.

### Audit of the slower layout measurements

The earlier full-layout comparison script had an asymmetric import boundary:
the control reconstructed the old fusion helper with `git show` and executed
its imports before conversion timing. The candidate imported RT-DETR on demand
inside initialization. Reproducing the latter import took 1.100 seconds. This
invalidates the old timings as a clean estimate of the fusion effect. It does
not explain every difference: pipeline time and unrelated table time also
varied. Do not attribute the remaining difference to background load without
controlled evidence.

An ablation with the model imports common to all variants measured the same
real four-page input batch in rotating order, with three forwards per variant
per round. Excluding the first diagnostic round, seven-round medians were:

| Transformation | Layout forward |
| --- | ---: |
| Existing 55 ResNet folds | 388.795 ms |
| Additional 42 BatchNorm folds | 382.849 ms |
| Additional twelve branch merges | 368.644 ms |

The normal four-page batch is unchanged; this is not a batching optimization.
Single-page replay also favored full fusion (113.089 to 105.888 ms). Separate
three-call stage probes put the hybrid encoder at 89.775 to 71.554 ms. Those
stage probes synchronize only in the isolated, single-thread replay and must
not be added to infer full-report savings. The backbone and decoder remain;
each branch merge still retains the expensive 3x3 convolution and removes the
cheaper 1x1 branch and addition. Operator counts are not compute-time weights.

Common-import full conversions returned 28.256/27.793 seconds for the control
and 26.809/26.699 seconds for full fusion. A later cold-import instrumented
pair returned 30.591 seconds for the control and 32.135 seconds for full fusion.
These conflicting full timings do not establish a stable speedup or regression.
The latter pair explicitly retained lazy model imports within initialization.
Both exports preserved all non-bbox fields and raw tags, 77 pages, 37 tables,
5877 cells, zero errors, and maximum bbox difference 0.0005951 PDF points.

Host tracing found substantial deferred work: in the candidate, BBox inference
returned after 0.314 seconds of host time, then BBox materialization waited
another 2.711 seconds. Decoder result materialization accumulated 3.354 seconds.
These waits include unfinished device computation, not just copying. Host-only
forward durations cannot identify individual GPU kernel costs. Guard-wait
durations overlap the other model's held intervals and are not extra wall time.
Evidence: `layout-ablation-replay-oct22-batch4.json`,
`layout-ablation-stages-oct22-batch4.json`, `layout-ablation-*/`, and
`full-time-ledger-clean-{2,baseline}/` in OntoScience's ignored `.tmp` directory.

#### Non-overlapping conversion ledger

A subsequent instrumented conversion completed in 28.180 seconds including
converter construction, or 28.034 seconds inside `convert`, including lazy
model initialization. This boundary excludes Python process startup, earlier
module imports, and diagnostic JSON export. It is a diagnostic sample, not a
new speedup claim. The same quality checks passed (77 pages, 37 tables, 5877
cells, identical non-bbox fields and tags, zero errors).

Partition the `convert` wall interval by initialization first, then mutually
exclusive model guards, then remaining wall time. Each interval is counted once:

| Exclusive interval | Seconds |
| --- | ---: |
| Layout initialization | 1.840 |
| Table initialization | 1.042 |
| Layout inference guard, excluding initialization | 9.172 |
| Table inference guard, excluding initialization | 11.403 |
| Outside initialization and inference guards | 4.578 |
| Total (unrounded) | 28.034 |

Within the table inference guard, decoder generator execution occupies 7.639
seconds, BBox host submission plus result materialization 2.914 seconds, and
other guarded work 0.850 seconds. Decoder result materialization (2.697 seconds)
is already inside the 7.639 seconds and must not be added again. Likewise,
layout/table guard waits (2.576/6.043 seconds) overlap active work and are not
additional wall costs. These are host intervals, not GPU kernel attribution;
queued device work can finish in a later materialization interval.

Of the 4.578 seconds outside guards, 2.396 seconds remain unattributed by the
current fine-grained spans. This is an explicit measurement gap, not proven
idle time. CPU matching/postprocessing and image preparation explain most of
the rest; page parsing also overlaps model work, so its cumulative 1.384
seconds must not be added to this exclusive ledger. The remaining investigation
must attribute layout operators and deferred BBox computation, and fill the
unattributed intervals before selecting further algorithm changes.

Evidence: `.tmp/full-time-ledger-final/{host-events,wall-ledger,timeline}.json`
and `.tmp/full-time-ledger-oct22-comparison.json` in the OntoScience checkout.
The timeline is a Chrome trace; the comparison contains the exclusive sweep
partition alongside inclusive counters. No device barriers were added to the
threaded conversion.

#### Refined ledger and isolated completion probes

Adding spans for PDF rendering, page resizing, cell selection, and output merge
reduced the uncovered interval to 0.710 seconds in a subsequent run. This run
took 30.058 seconds including converter construction, or 29.905 seconds inside
`convert`. It is a new diagnostic sample, not a speed comparison with 28.180
seconds above. The same process-startup/import/export exclusions apply.

| Exclusive interval | Seconds |
| --- | ---: |
| Converter construction outside `convert` | 0.153 |
| Layout and table initialization | 3.132 |
| Layout inference guard, excluding initialization | 9.890 |
| Table tag decoding, including result waits | 8.313 |
| Table BBox submission and result wait | 2.897 |
| Other guarded table work | 1.534 |
| Outside initialization and inference guards | 4.139 |
| Total, before rounding individual rows | 30.058 |

Of the 4.139 seconds outside guards, 3.428 seconds now intersect measured CPU
work and 0.710 seconds remain uncovered. Inclusive counters for rendering,
full-page resize, and PDF cell selection were 1.055, 1.143, and 0.492 seconds;
they overlap other intervals and are not additional wall time or promised
savings. There were 37 full-page resizes for 37 tables on 33 distinct pages.
Page 9 has five tables, so the current per-table predictor entry repeats its
page resize four extra times. Removing those four operations cannot eliminate
the whole 1.143-second resize cost. Cell token construction also deep-copies
each selected cell before deriving a scaled rectangle; its individual cost is
not separately established by these spans.

Quality comparison returned 77 pages, 37 tables, 5877 cells, zero conversion
errors, identical non-bbox document fields and raw tag multisets. Maximum bbox
delta against the reference remained 0.0005951 PDF points.

Sequential isolated probes used completion barriers only after threaded workers
had finished. A real four-page layout input, with full fusion, produced these
medians over three repeat diagnostic forwards after the first forward:

| Component | Completion-probed duration |
| --- | ---: |
| Backbone | 202.5 ms |
| Hybrid encoder | 72.9 ms |
| Detection decoder | 84.2 ms |
| Whole forward | 391.9 ms |

Component medians do not sum to the whole-forward median. The probes insert
barriers and do not establish a 77-page kernel-time distribution. They do show
that folding the hybrid encoder leaves substantial backbone and decoder work;
removed layer counts cannot predict overall savings.

A separate real 917-cell long-table BBox replay retained bitwise-equal class and
coordinate tensors. Over three repeats after the first probe, attention-score
construction took a median 271.9 ms, weighted image reduction 51.1 ms, softmax
11.1 ms, and output heads 18.3 ms. These are completion-probed compound
operations, not individual Metal kernel timings. The score path broadcasts
cell/image features before ReLU and projection; the final four-coordinate head
is not the dominant operation. This identifies a memory/computation target,
not a proven equivalent faster implementation or a whole-report speedup.

The longest table's actual recorded memory also replayed 1082 tags without
truncation. Original and instrumented decoders all matched its saved corrected
tag sequence. However, per-section barriers inflated the two instrumented
replays to 1.267/1.368 seconds, versus 0.866/0.714 seconds without those probes.
Those section totals must not be extrapolated to the 8.313-second full-report
decoder interval or used as an unperturbed GPU-kernel ranking. Self-attention,
cross-attention, projections/cache updates, output norms, and feed-forward work
all remain candidates; a device trace with less perturbation is still needed.
In particular, neither the result waits nor this replay prove that transfers
alone dominate decoding.

#### Decoder control-loop and operator follow-up

The next diagnostic replay used all 37 recorded encoder memories. It retained
887 verification blocks, 28,384 verification positions, and the saved corrected
tag sequences. The model has six decoder layers, eight heads, and head dimension
64. There are 9,180 generated tags, so generated tags account for about 32.3%
of verified positions. This ratio includes the final block's unused tail; it is
not an isolated measure of draft prediction accuracy.

A counterfactual replay recorded the actual proposal inputs, then ran the same
embeddings, positional encodings, cache updates, decoder, classifier, and argmax.
One mode read results after each of the 887 blocks. The other read results once
per table, at 37 boundaries. Every candidate prediction matched, including
rejected positions. Paired durations in seconds were 4.580/5.246,
9.351/6.025, and 8.761/8.085 (per-block/per-table). The second pair ran the
per-table mode first. These variable results do not establish a stable saving
or assign the full result-wait interval to transfer. Recorded inputs bypass the
real proposal/acceptance dependency: the per-table mode is a counterfactual,
not a usable decoder or a full-report performance result. Model initialization
and input capture are outside these diagnostic timings.

First-layer operator probes retained actual operand values, shapes, and strides
at positions 0, 511, and 1031 of the longest table. Each probe queued 40 calls
to one operator, then waited once. Three repeats followed the first probe.
At position 1031, the median completion cost per call was 0.210 ms for
self-attention, 0.091 ms for cross-attention, 0.058–0.094 ms for each of the six
linear operations, and 0.011–0.012 ms for each layer norm. These isolated
repeated-input costs exclude cache writes, residual additions, shape changes,
and inter-operator scheduling. They are not exclusive full-report GPU times and
must not be summed or extrapolated into a speedup claim. They identify both
long-context attention and repeated small linear operations as investigation
targets, rather than proving attention alone dominates.

The actual decoder executes 31,932 layer-internal linear calls, 10,644 attention
calls, and 15,966 layer-norm calls across these 887 blocks. These are Python/API
call counts, not Metal kernel counts. They exclude cache preparation and the
final classifier. Reducing rejected verification work and reducing submission
overhead therefore remain separate candidate optimizations.

Observed Q/K/V tensors have last-dimension stride 1. In the exact PyTorch source
for this runtime (commit `5c4886908584029761b579af026dcfb627c84070`), head
dimension 64 and query length 32 satisfy the native prefill attention predicates;
the strided KV views alone do not require contiguous copies in that path. This
is source/predicate evidence, not a captured device-kernel trace. Native MPS
operation statistics did not identify the attention kernel, so no stronger
backend claim is made.

Measurement limitations were also confirmed. This machine has no `xctrace`
utility, and `torch.profiler` exposes CPU activity only. Fine-grained MPS events
failed with an end-before-start timestamp error. The exact runtime implements
event timing with CPU completion callbacks and ends kernel coalescing when
recording an event; those samples are discarded for kernel attribution. Native
MPS profiler GPU durations can refer to shared command buffers, so adding those
per-operation durations can count the same device interval repeatedly.

Evidence in OntoScience's ignored `.tmp/`: `decoder-control-replay-oct22.json`,
`decoder-compound-cost-oct22.json`, `decoder-attention-route-oct22.json`, and their
same-name diagnostic scripts/logs. No runtime algorithm, concurrency, precision,
installed application, or acceptance baseline changed in this follow-up.

Local evidence: `full-time-ledger-refined/`,
`full-time-ledger-sampled/{stack-samples,gap-analysis}.json`,
`full-time-ledger-oct22-comparison.json`, `isolated-costs-oct22.json`, and
`decoder-costs-oct22.json` in OntoScience's ignored `.tmp/` directory. The trace
scripts changed diagnostics only. No inference optimization, concurrency,
weights, precision, installed application, or acceptance baseline changed in
this follow-up. The 7–14 second target remains unmet.

The separate causal row-boundary draft experiment reduced verification calls
from 887 to 822 and retained bitwise-equal tags/accepted hidden states. Later
replay rounds were nevertheless slower. It is not included in the source
algorithm. CPU input embedding also showed no stable benefit and is not selected.

Local diagnostic evidence lives in OntoScience's ignored `.tmp` directory:
`decoder-mask-full-*`, `layout-reparam-full-*`, `layout-reparam-cpu-fold-*`,
`layout-reparam-*-quality.json`, `layout-fold-init-oct22.json`, and
`decoder-row-replay-oct22.json`. No weights, precision, concurrency, cross-table
batching, product acceptance baseline, or installed application changed.
The 7–14 second complete-report target is not achieved.

## Remaining work and selected changes

### Bounded final decoder experiment: neither candidate selected

A final experiment tested two candidates without changing concurrency, table
batching, model weights, or float32 tensor precision. Both remained isolated
diagnostic overrides; neither changed the runtime implementation.

The confidence-limited draft used only accepted history. It multiplied online
transition confidence and stopped a proposal chain below 0.25, while retaining
at least eight verification positions. Across all 37 recorded table memories,
verification positions fell from 28,384 to 13,340. Verification calls increased
from 887 to 1224. All three isolated decoder replays were slower: baseline
6.308/6.413/5.466 seconds versus candidate 9.278/9.042/9.419 seconds. Corrected
tags matched; maximum accepted-hidden-state difference was 0.00005174. Fewer
verified positions did not compensate for more sequential model calls. This
candidate is rejected, not evidence against every possible adaptive draft.

The fusion candidate combined residual addition and layer normalization into
one Metal kernel at each of the three norm sites per layer. It retained float32
but changed reduction order. It kept 887 calls and 28,384 positions. Isolated
decoder median time fell from 6.308 to 5.742 seconds, with one of three paired
replays slower. All tags matched; maximum accepted-hidden-state difference was
0.00004291. This local result was insufficient to select the candidate.

Six fresh-process complete conversions then tested fusion in the order baseline,
fused, fused, baseline, baseline, fused. Timings started before converter
construction, included lazy model initialization and first-use shader compilation,
and ended after conversion. Process startup, imports, diagnostic wrapper setup,
and JSON export remained outside the timing. No warmup conversion was run.

| Pair | Baseline seconds | Fused seconds |
| --- | ---: | ---: |
| 1 | 30.432 | 28.031 |
| 2, fused first | 27.927 | 29.663 |
| 3 | 26.901 | 27.540 |
| Median | 27.927 | 28.031 |

The complete-report median increased by 0.37%; one pair improved and two
regressed. Median table-model time was 9.032 seconds versus 9.082 seconds.
These samples do not prove fusion is inherently slower, but they do not
establish a stable end-to-end gain. The machine was not an isolated benchmark
host. The result does not justify adding a platform-specific runtime kernel.

All six conversions returned 77 pages, 37 tables, 5877 cells, no conversion
errors, identical non-bbox document fields, and identical multisets of complete
table tag sequences. Baseline coordinates repeated exactly. Fused coordinates
differed by at most 0.0055846 PDF points from the first baseline; no quality
tolerance or acceptance baseline was changed to admit this difference.

Neither candidate is selected. Per the agreed stopping condition, this decoder
optimization investigation stops here; it does not establish that further
optimization is mathematically impossible. The 7–14 second complete-report goal
is not achieved. Existing runtime code and the installed application remain
unchanged. This is diagnostic evidence only, not product acceptance; no new
Acceptance Case or acceptance-baseline promotion is required.

Local evidence in OntoScience's ignored `.tmp/`: `decoder-final-candidates-oct22.json`,
`decoder-final-comparison-oct22.json`, `decoder-final-full-oct22-*/`, and the
corresponding candidate, full-conversion, and comparison scripts. Synthetic
kernel checks also exposed reduction-order sensitivity for large-offset inputs;
their looser diagnostic tolerance was not applied to document acceptance.

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
