# Legacy comparison

The public single-point evaluator is
`SinglePointEvalOpen.legacy_compatible_runner`. It carries the established
`SingleHopEval/ask_vllm.py` flow, with a flat QA adapter at
`read_question_info`. Video discovery, on-demand frame sampling, face boxes,
subtitle rendering, image grouping, threaded inference, retries, fallback,
atomic JSON output, and historical scoring remain in this path. There is no
second single-point evaluator with different scoring or result rules.

Conan uses `SinglePointEvalOpen.conan_runner` because it has a separate model
backend. That launcher adapts the same flat QA input and invokes Conan-Eval;
it does not replace the compatibility runner for the other models.

## Other evaluation tasks

| Task | Historical logic retained | New code around it |
| --- | --- | --- |
| Multi-point | `MultiPointEvalOpen/legacy_evaluator.py` is copied from `MultiPointEval/ask_multi_point.py`; the optional OpenAI import is delayed until client creation, and frame lookup now prefers the public sampler's drama directory before falling back to historical aliases. Prompt, chronological scene ordering, extraction, fallback and scoring remain there. | `qa_format.py` validates the public list format and maps TTC-2/TTC-3/CECI to the old `B_order`/`B_sort`/`D_same` groups. `prepare_frames.py` reuses the single-point sampler for each distinct L3 window; `run_eval.py` provides the CLI. |
| Multi-hop and paired sources | The same copied multi-scene evaluator handles two L3 scenes in story order and keeps the original scene labels. | `MultiHopEvalOpen/qa_format.py` maps one public multi-hop row to the historical `multi_hop` input. Its `source_q1` and `source_q2` modes reuse the same two scenes and share a stable pair ID. The supplied rows have no source-question IDs, so the paired-source outputs are per pair and are **not** the historical deduplicated-source metric. |
| Six ablations | `AblationEvalOpen/ask_vllm.py` copies the historical ablation evaluator. The six mode branches and their evaluation logic remain in the copy. | `qa_adapter.py` reshapes the flat single-point QA to the old nested format. `run_eval.py` passes the historical `--parallel` mode and mode-specific subtitle flags. `prepare_local_inputs.py` relocates the historical gallery/input preparation tool. Embedded credentials, machine-specific checkpoint defaults and bundled-font assumptions are removed from the copy. |
| Conan | `SinglePointEvalOpen/conan_runner.py` calls the existing Conan-Eval backend with adapted QA. | The Conan model's multi-step inference implementation and weights are **not** included in this package; they must be supplied separately. |

For the single-point compatibility runner, the new flat QA adapter, safer
image-root resolution and model-name configuration are input/runtime changes.
When `decord` is unavailable, its new FFmpeg fallback seeks using frame index
divided by FPS; on variable-frame-rate video this may select a different image
from `decord`. The system-font fallback may also render Chinese subtitles
differently unless `EVAL_FONT_PATH` points to the historical font. Those are
material limits on exact visual parity; they are not scoring-rule changes.
The public multi-point and multi-hop adapters use content-derived question IDs
and separate output directories, so their result files are not a drop-in
resume continuation of historical runs with different IDs.

## Public CLI normalization (2026-10-31)

`dramavqa validate`, `prepare`, and `run` dispatch to the existing engines.
They select a precise number of questions before adaptation, so a clip with
multiple questions no longer makes `--limit` exceed the requested question count.
Legacy module-level clip limits remain compatible.

Single-point and ablation local clients now use `OPENAI_BASE_URL` and
`OPENAI_API_KEY`; explicit proxy settings are preserved and TLS verification
is enabled. The ablation image-root check uses path components, matching
the single-point runner. CLI help is English. Gallery preparation is included
in the public preparation stage. Default model output budgets remain 1000
tokens for API evaluation and 4000 for Conan.

New manifests guard against mixing inputs or model settings in one result
directory. Completeness checks reject missing frames, skipped answers and
random fallback as a successful smoke run; they do not change historical
full prompt text, answer regexes, guesses or accuracy calculations. Other prompt modes have been removed; the full prompt retains thinking and non-thinking variants. The public summary
also reports exact reference matches. Model output format failures remain
visible and must not be presented as successful inference.
