# DramaVQA Evaluation

Run all commands from the repository root.

## Environment installation

Requires Python 3.10+ and FFmpeg/ffprobe. On Ubuntu/Debian, install FFmpeg
with `sudo apt install ffmpeg`.

```bash
python -m pip install ./evaluate huggingface_hub
```

Chinese and English Noto fonts are bundled and loaded automatically for labels
and subtitles.

## Download data

Download videos and metadata from
[Hugging Face](https://huggingface.co/datasets/198808xc/drama_bench_data).
Download one drama (replace `the_knockout` as needed):

```bash
hf download 198808xc/drama_bench_data --repo-type dataset \
  --include 'the_knockout/*' --local-dir data/drama_bench_data
```

Download the complete dataset:

```bash
hf download 198808xc/drama_bench_data --repo-type dataset \
  --local-dir data/drama_bench_data
```

The downloaded media layout is:

```text
data/drama_bench_data/<drama>/
├── video_data/0001.mp4
├── face_video_data/0001.mp4
├── face_data/0001.json
└── subtitle_data/0001.json
```

Use this download directory as `--video-root` in the preparation commands below.

## Evaluation overview

- **Main benchmark (single-point):** evaluate detailed understanding of one scene
  with L3–L6 context windows.
- **Multi-point:** evaluate relationships across multiple scenes.
- **Multi-hop:** combine information from two scenes to answer a reasoning question.

Each task uses three commands:

1. `validate`: check question format and selected question count, without calling a model.
2. `prepare`: extract the required frames and subtitle inputs.
3. `run`: send prepared inputs to the model service and save predictions and scores.

## Quick start

The examples use ten questions and eight frames. Omit `--limit` from all steps
to evaluate all questions, and set `--sample-frames` to the required frame count.
Start an OpenAI-compatible vision-language model service before `run`. Replace
`my-model` with its served model ID and `--base-url` with its API endpoint. Set
`OPENAI_API_KEY` if authentication is required.

### Main benchmark (single-point)

```bash
# Step 1: validate questions.
dramavqa validate --task single-point --drama the_knockout --limit 10

# Step 2: prepare frames and subtitles.
dramavqa prepare --task single-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/single-point

# Step 3: run model evaluation.
dramavqa run --task single-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/single-point \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

The default question set is Mini. Add `--subset Bench_Data` to all three
commands for the full benchmark. Change `--level` to L4, L5, or L6 for other
context windows.

**All local dramas:** use `--drama all` to process downloaded drama folders
under `--video-root` that have matching questions in the selected set.
`--limit 10` selects ten questions **per drama**.

```bash
dramavqa validate --task single-point --drama all --limit 10 \
  --video-root data/drama_bench_data --output-dir runs/single-point-all

dramavqa prepare --task single-point --drama all --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/single-point-all

dramavqa run --task single-point --drama all --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/single-point-all \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

The `run` command reuses the drama list saved by `prepare`, so `--video-root`
can be omitted at inference. Outputs are separated under `<output-dir>/<drama>/`.
The same workflow applies to multi-point and multi-hop.

### Multi-point

```bash
# Step 1: validate questions.
dramavqa validate --task multi-point --drama the_knockout --limit 10

# Step 2: prepare all source-scene inputs.
dramavqa prepare --task multi-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/multi-point

# Step 3: run model evaluation.
dramavqa run --task multi-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/multi-point \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

### Multi-hop

```bash
# Step 1: validate questions.
dramavqa validate --task multi-hop --drama the_knockout --limit 10

# Step 2: prepare both source-scene inputs.
dramavqa prepare --task multi-hop --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/multi-hop

# Step 3: run model evaluation.
dramavqa run --task multi-hop --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/multi-hop \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

### Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--task` | Required | `single-point`, `multi-point`, or `multi-hop`. |
| `--drama` | Required | Drama name, e.g. `the_knockout`; `all` selects local dramas with matching QA. |
| `--video-root` | Required for `prepare` and batch `validate` | Media download directory, e.g. `data/drama_bench_data`. |
| `--level` | `L3` | Single-point context: L3–L6. Other tasks use L3. |
| `--sample-frames` | `8` | Frames sampled per context window. |
| `--output-dir` | Required for `prepare` and `run` | Use a separate directory for each task, model, and setting. |
| `--model` | Required for `run` | Served model ID, e.g. `my-model`. |
| `--base-url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible API endpoint; can also be set with `OPENAI_BASE_URL`. |

Keep question selection, context, sampling, subtitle, and output settings
consistent between `prepare` and `run`. See `dramavqa run --help` for all options.

## Results

Outputs are saved under `--output-dir`:

- `inputs/`: selected questions.
- `images/`: prepared frames.
- `results/`: predictions and evaluation results.

Check `frames_summary.json` for frame preparation and `summary.json` for
completion and valid-answer counts. Predictions and scoring outputs are saved
in `results/`.

Batch summaries are saved in the parent output directory as
`batch_validate_summary.json`, `batch_prepare_summary.json`, and
`batch_run_summary.json` after their respective stages.
