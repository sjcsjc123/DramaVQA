# DramaVQA evaluation

A consistent command line for single-point, multi-point, multi-hop, paired-source
and ablation evaluation. Existing adapters preserve benchmark prompts, scene
ordering, answer extraction and scoring. Use the public `dramavqa` interface
for new runs; the historical Python modules remain available for compatibility.

## Repository scope

This repository is an independently installable evaluation client. It contains
all four evaluation task packages, the common CLI, synthetic test fixtures and
documentation. It does not require the original workspace or deployment scripts.
A model service, benchmark data, source media and a Chinese-capable rendering
font must be supplied separately. The optional Conan backend also requires a
compatible external Conan-Eval checkout and checkpoint.

The `evaluate/` directory contains `pyproject.toml`, `dramavqa/`, the four
`*EvalOpen/` packages, `tests/` and `docs/`. No datasets, videos, checkpoints,
private configuration, runtime environments or experiment outputs are bundled.
The `legacy` install extra is retained as an empty compatibility alias; standard
evaluation dependencies are now installed by default.

## Install

Requires Linux, Python 3.10+, FFmpeg/ffprobe and a model server. Install from the
`evaluate/` directory:

```bash
# From the repository root:
cd evaluate
python -m pip install '.[test]'
python -m dramavqa --help
python -m pytest -q tests
```

`dramavqa` and `python -m dramavqa` are equivalent. All public arguments use
hyphens. API credentials come from `OPENAI_API_KEY`; never put them in scripts,
configuration committed to Git, or command arguments. `OPENAI_BASE_URL` sets
the default endpoint; `--base-url` overrides it. Standard proxy and CA environment
variables are respected.

## One workflow for every task

| Task | `--task` | QA input | Context |
| --- | --- | --- | --- |
| Single point | `single-point` | Bench_Data, mini benchmark, or training JSON | L2–L5 |
| Multiple points | `multi-point` | Multi-Point JSON | L3 |
| Multiple hops | `multi-hop` | Multi-Hop JSON | L3 |
| Paired first/second source | `source-q1`, `source-q2` | Same Multi-Hop JSON | Same two L3 scenes |
| Single-point ablation | `ablation` | Flat single-point JSON | L2–L5; also requires `--mode` |

Each command accepts `--qa`, `--drama`, `--limit`, `--output-dir`,
`--sample-frames`, `--target-width` and `--level`. `--limit 10` means **ten
questions**, even when a clip contains several questions. Zero selects all.
The default eight frames are intended for quick checks; explicitly set the
published frame count (for example 256) for full benchmark experiments.
Selection follows file order, so it is reproducible but not statistically
representative. Validate the whole dataset before selecting a subset.

```bash
# 1. Validate the input without a model.
dramavqa validate --task single-point --qa /data/qa/kuang_biao.json \
  --drama kuang_biao --limit 10

# 2. Generate frames, face annotations and text subtitles.
dramavqa prepare --task single-point --qa /data/qa/kuang_biao.json \
  --drama kuang_biao --limit 10 --level L2 --sample-frames 8 \
  --video-root /data/Release_Drama --output-dir /data/runs/single-point

# 3. Evaluate those exact ten questions against a real model server.
dramavqa run --task single-point --qa /data/qa/kuang_biao.json \
  --drama kuang_biao --limit 10 --level L2 --sample-frames 8 \
  --output-dir /data/runs/single-point --model my-model \
  --base-url http://127.0.0.1:8000/v1
```

The two stages must use identical input, limit, context, frame and subtitle
settings. Preparation checks every selected window for its complete frame set.
Evaluation writes `summary.json` and exits nonzero for missing, skipped,
randomly guessed or truncated answers. A response ending at the token limit
is incomplete even if an option was extracted from its unfinished reasoning;
the last retry determines truncation status. An incorrect model answer still counts as completed
inference; completeness and accuracy are different measurements.

Use a separate output directory for each task, model and setting. The CLI
writes the selected QA under `inputs/`, fingerprints its content and rendering
settings, and rejects accidental reuse with different settings. Original
result formats remain under `results/`; `summary.json` provides a common
completion report. Existing compatible results can be resumed. If a compatible API omits token
usage, answers are still saved and token usage remains `null` (unknown); the
report does not fabricate a zero-token total.

For multi-point or multi-hop, change `--task` and `--qa` in the same three
commands, and use `--level L3`. Multi-scene preparation always writes
`OUTPUT_DIR/images`; evaluation can reuse another frame tree with `--image-root`.
It preserves chronological concatenation and source labels. Paired-source
modes retain all pairs, not the historical deduplicated-source metric.

## Full prompt and thinking

Only `full` is supported. API single-point and ablation evaluation retain the
original two variants:

- `--thinking off` (default): task description, answer format and example.
- `--thinking on`: reasoning instructions ending with `Final Answer: <letter>`.

`--prompt-mode full` is optional. Other prompt modes have been removed.
Multi-scene and Conan backends retain their own original prompts; they reject
`--thinking on` because they do not implement this prompt switch.

This is a **prompt** choice. The independent model/API option `ENABLE_THINKING`
(`auto`, `on`, `off`; default `auto`) controls model-side thinking where the
backend supports it. The CLI explicitly sets `THINK_MODE` for the child runner;
legacy direct runner commands may still use `THINK_MODE=0` or `THINK_MODE=1`.
Both settings are recorded in `run_manifest.json`. Use separate output directories.

The original full prompt includes a literal `<letter>` placeholder. Some models
copy it instead of choosing an option. The completion report exposes failed
extraction and historical random fallback, and does not count fallback as a
clean model answer. Prompt text and extraction rules have not been rewritten.

## Source data and rendering

Expected layout:

```text
Release_Drama/<drama>/
  video_data/version.json
  video_data/<active-version>/0001.mp4
  face_data/version.json
  face_data/<active-version>/0001.json
  subtitle_data/version.json
  subtitle_data/<active-version>/0001.json
```

The version manifest selects the active directory; an available subdirectory
is used when no matching entry exists. Seasonal drama mappings are in
`DRAMA_NAME_MAP`. Download every episode touched by the chosen context window,
including intermediate episodes. L4/L5 can span many episodes.

`--subtitles text` is the default: face-box frames plus separate subtitle text.
Use `--subtitles burn` or `none` for single-point evaluation. Multi-scene tasks
use text subtitles. Face and subtitle metadata are optional for a basic run,
but necessary to reproduce their benchmark input policy. Set `EVAL_FONT_PATH`
to a licensed Chinese-capable font for Chinese labels/subtitles; no proprietary
font is bundled. `--use-rendered-video` reads pre-rendered face videos.
Install `decord` for native decoding, or use the FFmpeg fallback. Its time-based
seeking can differ for variable-frame-rate videos; validate frame identity
before reporting reproducibility claims.

## Ablations

Use `--task ablation --mode MODE` with the same prepare/run workflow. The mode
determines rendering and subtitle policy:

| Mode | Input policy |
| --- | --- |
| `exp1_burn` | Burned subtitles |
| `exp2_nosub` | Text subtitles |
| `exp3_gallery` | Face gallery, raw frames and text subtitles |
| `exp4_speaker` | Face-box frames, text subtitles and speaker labels |
| `exp5_nosub` | Face-box frames without subtitles |
| `exp6_langonly` | Question and options only |

For `exp4_speaker`, provide `--speaker-root` pointing to the parent of
`<drama>/speaker_data`. Gallery mode prepares and validates its face gallery automatically during
`dramavqa prepare`; provide the same `--video-root` again during `run` so
cache provenance can be verified.
Language-only evaluation requires no frame generation. Use distinct output
directories for every ablation mode.

## Conan backend

Conan retains its own model implementation and stepwise inference. Prepare
frames with the same public command, then run:

```bash
dramavqa run --task single-point --backend conan \
  --qa /data/qa/kuang_biao.json --drama kuang_biao --limit 10 \
  --level L2 --sample-frames 8 --output-dir /data/runs/conan \
  --image-root /data/runs/single-point/images \
  --conan-eval /models/Conan/Conan-Eval --model-path /models/Conan-7B \
  --python /envs/Conan/bin/python --gpu-ids 0 --max-tokens 4000
```

The external Conan checkout must support `CONAN_USE_DRAMA_QA` and the path
variables in `SinglePointEvalOpen/conan_runner.py`. It and its weights are not
part of this package. Do not assume a copied Conda environment is relocatable:
verify Python, shared libraries, Torch/CUDA and model imports after restoration.

## Development and publication

- [Advanced/legacy commands](docs/ADVANCED.md)
- [Differences from historical evaluators](COMPARISON.md)

Local model trial reports, deployment utilities and machine-specific environments are excluded from this repository. See RELEASE_NOTES.md for the release validation scope.

Comments and docstrings in this package are English. Chinese benchmark text,
role-name handling, answer regexes and some legacy runtime messages remain
unchanged to avoid altering evaluation semantics.

This code follows the repository MIT license, reproduced in LICENSE for
standalone distribution. Dataset annotations, media, fonts and external model
implementations retain their own terms; the code license does not grant
redistribution rights to those materials. Exclude datasets, model weights,
runtime environments, private paths/configuration and generated results from
the public repository. Build from the `evaluate/` directory:

```bash
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```
