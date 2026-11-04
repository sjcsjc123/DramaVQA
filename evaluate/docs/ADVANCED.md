# Advanced and legacy entry points

For new runs use the [public CLI](../README.md). This reference preserves
the original module interfaces, whose `--limit` may count clips.


This directory contains the open evaluation entry points for the current
single-point, multi-point, multi-hop and paired-source QA files. It is
intentionally separate from the historical evaluation scripts;
the original `SingleHopEval` and `MultiPointEval` directories
are not modified by this version.
After installing this repository, its
three task packages are available through the documented `python -m ...`
commands.
The evaluator follows the repository MIT license, reproduced in LICENSE.
External data, media and model implementations retain their own terms.

The three task packages are `SinglePointEvalOpen`, `MultiPointEvalOpen`, and
`MultiHopEvalOpen`. The historical multi-scene evaluator is copied into this directory;
the new adapters change their QA input shape without changing scene ordering,
answer extraction, or scoring. `SingleHopEval` is the name of an old directory
for the single-point benchmark. The paired `source_q1`/`source_q2` controls
below are a separate task.

## Paths for this GitHub checkout

All examples here run from the repository root after installing `./evaluate`.
Question JSON files live in `DramaVQA/`; video/metadata downloads are separate
under `data/drama_bench_data/`, as described in the project README.
The public CLI can discover question paths or accept `--data-root`; the legacy
module commands below still require explicit input paths.

## Multi-point and multi-hop QA

The supplied DramaVQA archive contains 3,200 multi-point questions (12 files) and 1,200
multi-hop questions (12 files). Multi-point dimensions map to the old evaluator
types `B_order`, `B_sort`, and `D_same`. The multi-hop rows also contain two
source questions with options, answers, and L3 windows. The modes `source_q1`
and `source_q2` evaluate each source question with the **same two scenes** as
its paired multi-hop question. The public rows have no `q1_id` or `q2_id`, so
these modes retain all 1,200 pairs rather than claiming the historical
deduplicated-source metric.

Validate the tracked question files without downloading media:

```bash
python -m pip install ./evaluate
python -m MultiPointEvalOpen.run_eval --qa DramaVQA/Multi-Point/friends.json --drama friends --validate-only
python -m MultiHopEvalOpen.run_eval --qa DramaVQA/Multi-Hop/friends.json --drama friends --validate-only
```

To start from source videos, the frame preparation command reuses the
single-point compatibility sampler. It writes one L3 frame directory for each
distinct source window, with face boxes when face data is available and text
subtitles in `subtitles.json` when subtitle data is available. The source video
layout is documented below. Set `--kind multi_hop` for the multi-hop file.

```bash
python -m MultiPointEvalOpen.prepare_frames \
  --qa DramaVQA/Multi-Point/friends.json --kind multi_point --drama friends \
  --video-root data/drama_bench_data --output-root runs/legacy/multi_point \
  --sample-frames 256 --target-width 392

python -m MultiPointEvalOpen.run_eval \
  --qa DramaVQA/Multi-Point/friends.json --drama friends \
  --image-root runs/legacy/multi_point/images/L3/256/results.w392_nosub/img_data \
  --output-dir runs/legacy/multi_point/results/friends \
  --model my-served-model --base-url http://localhost:8000/v1
```

Use `MultiHopEvalOpen.run_eval` with the same image layout for multi-hop. Give
each mode a distinct output directory so its historical `results.json` and
`raw_responses.jsonl` do not overwrite another mode:

```bash
python -m MultiPointEvalOpen.prepare_frames \
  --qa DramaVQA/Multi-Hop/friends.json --kind multi_hop --drama friends \
  --video-root data/drama_bench_data --output-root runs/legacy/multi_hop
python -m MultiHopEvalOpen.run_eval \
  --qa DramaVQA/Multi-Hop/friends.json --drama friends --mode multi_hop \
  --image-root runs/legacy/multi_hop/images/L3/256/results.w392_nosub/img_data \
  --output-dir runs/legacy/multi_hop/results/multi_hop/friends \
  --model my-served-model
python -m MultiHopEvalOpen.run_eval \
  --qa DramaVQA/Multi-Hop/friends.json --drama friends --mode source_q2 \
  --image-root runs/legacy/multi_hop/images/L3/256/results.w392_nosub/img_data \
  --output-dir runs/legacy/multi_hop/results/source_q2/friends \
  --model my-served-model
```

Both runners read `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the environment
when set; local defaults are `http://localhost:8000/v1` and `EMPTY`. The
multi-scene evaluator preserves its chronological scene concatenation and
original scene labels. Its existing random-choice fallback after an
unextractable answer also remains unchanged.

## Current QA format

The current single-point files are flat JSON objects: each top-level key is a clip id and
`Q&A` is a list. A clip can contain several independent questions. The validator checks each item and its L2--L5 windows. The compatibility
runner retains the historical internal question IDs. It does not interpret or
convert multi-hop data.
Clips with an empty `Q&A` list are retained in validation and contribute zero
question records; this matches the supplied training data.

## Single-point evaluation

`SinglePointEvalOpen.legacy_compatible_runner` is the single-point evaluation
entry point. It adapts the current flat QA file to the historical internal
shape and keeps the existing video scan, frame sampling and rendering,
inference, answer extraction, scoring, and output format. Install the package
and run frame generation followed by inference:

```bash
python -m pip install ./evaluate

# Stage 1: start with QA and source videos; generate frames.
python -m SinglePointEvalOpen.legacy_compatible_runner \
  --drama the_knockout --question_path DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --video_path data/drama_bench_data --image_path runs/legacy/images \
  --sample_frames 8 --level L3 --model_name my-vl-model \
  --serving_mode local --only_gen_img --limit 10

# Stage 2: start an OpenAI-compatible model server, then evaluate.
python -m SinglePointEvalOpen.legacy_compatible_runner \
  --drama the_knockout --question_path DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --image_path runs/legacy/images --result_path runs/legacy/results \
  --sample_frames 8 --level L3 --model_name my-vl-model \
  --serving_mode local --port 8000 --use_render_info --limit 10
```

`--limit 10` provides a small end-to-end check before a larger run. The result
uses the historical `q_a_dict.json` and `raw_responses.jsonl` layout. The two
stages must use the same frame count, rendering mode, and image path.

The current video layout is `data/drama_bench_data/<drama>/video_data/0001.mp4`
(or `.mkv`), with face videos in `face_video_data/0001.mp4` and episode JSON
metadata directly in `face_data/` and `subtitle_data/`. No `version.json` is
required. Legacy version manifests and subdirectories remain supported.
Every drama uses one directory named after its QA JSON stem. English-language
dramas use continuous episode numbering across seasons: after `0008` comes
`0009`, including videos, face/subtitle metadata, and speaker metadata.
Season-specific directories such as `breaking_bad_1/` are no longer used.
`face_data` and `subtitle_data` are optional for a basic run, but are needed
to reproduce face-box and subtitle rendering from the benchmark dataset.
When using pre-rendered face videos, select `--use_rendered_video` in stage 1.
The compatibility runner reads `OPENAI_BASE_URL`, falling back to
`http://localhost:<port>/v1`. By default `--model_name` must match the server's
`--served-model-name`; `VLLM_SERVED_MODEL_NAME` overrides the request model.
Use `--no_render_subtitle` in both stages when no subtitle annotations should
be rendered.

Run `python -m SinglePointEvalOpen.legacy_compatible_runner --help` for all
arguments. The runner needs the declared package dependencies plus `ffmpeg`
and `ffprobe` when `decord` is unavailable. Face-box and subtitle rendering
require Pillow built with FreeType. Set `EVAL_FONT_PATH` to the historical
Chinese TrueType font for the same rendered text as the old environment;
a system DejaVu font is used only as a portability fallback.

Install optional test dependencies and run:

```bash
python -m pip install './evaluate[test]'
python -m pytest -q evaluate/tests
```

The FFmpeg fallback seeks by frame index divided by FPS. For variable-frame-rate
sources, install `decord` or check sampled frame identity before scoring.

Conan's existing evaluator expects `category -> clip -> Q&A dict`, while the
current QA is `clip -> Q&A list`. Convert the input without editing Conan:

```bash
python -m SinglePointEvalOpen.conan_qa_adapter \
  DramaVQA/Bench_Data_mini_2000/the_knockout.json /path/to/conan_qa/the_knockout.json
```

The converter preserves each question and its L2--L5 windows. The model's
existing inference code is in `models/Conan/Conan-Eval/main_singlehop.py`,
with weights in `models/Conan/Conan-7B` and Python in `conda/Conan`. The new
launcher below converts the current QA and invokes that evaluator directly;
it does not import or call `SingleHopEval`. The Conan backend must support the
`CONAN_USE_DRAMA_QA` and related path environment variables used here.
Conan-Eval is a separate model implementation and is not bundled in
this repository; reproduce its run with the compatible Conan-Eval checkout and
model weights supplied through `--conan-eval` and `--model-path`.

```bash
python -m SinglePointEvalOpen.conan_runner \
  --qa DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --image-root runs/legacy/images \
  --output-root runs/legacy/conan_results \
  --conan-eval /path/to/models/Conan/Conan-Eval \
  --model-path /path/to/models/Conan/Conan-7B \
  --python /path/to/conda/Conan/bin/python \
  --drama the_knockout --level L3 --sample-frames 8 \
  --image-subdir results.w392_nosub --gpu-ids 0 --num-processes 1
```

Use `--dry-run` to validate paths, QA conversion and the generated command
without loading the model. The launcher preserves Conan's step mode, model
parameters, resume behavior and result format. Frame generation remains the
first stage above; point `--image-root` to its output.
