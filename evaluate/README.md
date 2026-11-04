# DramaVQA Evaluation

Run the commands below from the repository root. Requires Python 3.10+,
FFmpeg/ffprobe, and a running OpenAI-compatible vision-language model service.

## Install and download data

Download videos and metadata from
[Hugging Face](https://huggingface.co/datasets/198808xc/drama_bench_data).
Replace `the_knockout` in the command below to download another drama.

```bash
python -m pip install ./evaluate huggingface_hub
hf download 198808xc/drama_bench_data --repo-type dataset \
  --include 'the_knockout/*' --local-dir data/drama_bench_data
```

After downloading, the local media layout is:

```text
data/drama_bench_data/<drama>/
├── video_data/0001.mp4
├── face_video_data/0001.mp4
├── face_data/0001.json
└── subtitle_data/0001.json
```

The `prepare` commands below use `--video-root data/drama_bench_data` to read
media from this download directory. Replace that path if you save data elsewhere.

## Evaluate

Single-point evaluates details within one scene; multi-point evaluates relationships
across scenes; single-hop controls test the two source questions separately;
multi-hop combines information from two scenes for reasoning.
Follow the commands for your task in three steps:

1. `validate`: check the question format and selected question count; no model is called.
2. `prepare`: extract the frames and subtitle inputs needed by those questions.
3. `run`: send the prepared inputs to your model service and save predictions and scores.

The examples use ten questions and eight frames; omit `--limit` in all steps
to evaluate all questions. Replace `my-model` and the API
address with your model service. Set `OPENAI_API_KEY` if authentication is
required.

### Font download and configuration

For Chinese labels/subtitles, use [Noto Sans CJK Regular (TTC)](https://github.com/notofonts/noto-cjk/blob/main/Sans/OTC/NotoSansCJK-Regular.ttc).
For English-only labels/subtitles, use [Noto Sans Regular (TTF)](https://github.com/notofonts/noto-fonts/blob/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf).
The evaluator accepts both TTC and TTF files; English does not require a TTC.
Use the CJK font for mixed Chinese/English text. An English-only font does not
provide Chinese glyphs.

```bash
mkdir -p fonts

# Chinese or mixed Chinese/English.
curl -fL https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTC/NotoSansCJK-Regular.ttc \
  -o fonts/NotoSansCJK-Regular.ttc
export EVAL_FONT_PATH="$(pwd)/fonts/NotoSansCJK-Regular.ttc"

# English only: use this setting instead.
curl -fL https://raw.githubusercontent.com/notofonts/noto-fonts/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf \
  -o fonts/NotoSans-Regular.ttf
export EVAL_FONT_PATH="$(pwd)/fonts/NotoSans-Regular.ttf"
```

Choose one font setting before `prepare`; keep it for `run` if it renders labels
or subtitles. Font files are downloaded separately and are not bundled with the code.

### Single-point

**Step 1 — Validate questions.** The default input is
`DramaVQA/Bench_Data_mini_2000/the_knockout.json`.

```bash
dramavqa validate --task single-point --drama the_knockout --limit 10
```

**Step 2 — Prepare video frames and subtitles.**

```bash
dramavqa prepare --task single-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/single-point
```

**Step 3 — Run model evaluation.**

```bash
dramavqa run --task single-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/single-point \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

### Multi-point

The input is `DramaVQA/Multi-Point/the_knockout.json`. Each question uses its
source scenes with L3 context windows.

```bash
# Step 1: validate questions.
dramavqa validate --task multi-point --drama the_knockout --limit 10

# Step 2: prepare all source-scene frames and subtitles.
dramavqa prepare --task multi-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/multi-point

# Step 3: run model evaluation.
dramavqa run --task multi-point --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/multi-point \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

### Single-hop source-question controls

Evaluate the first or second source question of each multi-hop pair with
`source-q1` or `source-q2`. Both controls use the same two L3 scenes as the
multi-hop question. They read `DramaVQA/Multi-Hop/the_knockout.json`; results
are counted per pair, without deduplicating repeated source questions.
These commands are paired-source controls, not a separate single-hop dataset.

**First source question:**

```bash
# Step 1: validate questions.
dramavqa validate --task source-q1 --drama the_knockout --limit 10

# Step 2: prepare the paired scene inputs.
dramavqa prepare --task source-q1 --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/source-q1

# Step 3: evaluate the first source question.
dramavqa run --task source-q1 --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/source-q1 \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

**Second source question:**

```bash
# Step 1: validate questions.
dramavqa validate --task source-q2 --drama the_knockout --limit 10

# Step 2: prepare the paired scene inputs.
dramavqa prepare --task source-q2 --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --video-root data/drama_bench_data \
  --output-dir runs/source-q2

# Step 3: evaluate the second source question.
dramavqa run --task source-q2 --drama the_knockout --limit 10 \
  --level L3 --sample-frames 8 --output-dir runs/source-q2 \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

### Multi-hop

The input is `DramaVQA/Multi-Hop/the_knockout.json`. Evaluate the combined
multi-hop question using its two source scenes.

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

### Common parameters

| Parameter | Description | Example / default |
| --- | --- | --- |
| `--task` | Evaluation task. | `single-point`, `multi-point`, `source-q1`, `source-q2`, or `multi-hop` |
| `--drama` | Drama name matching the QA filename stem and media directory. | `the_knockout` |
| `--video-root` | Media download directory used by `prepare`; contains the drama folders. | `data/drama_bench_data` |
| `--level` | Video context window. Single-point supports L3–L6; other tasks use L3. | Default: `L3` |
| `--sample-frames` | Frames sampled per context window. | Default: `8`; set the frame count required by your experiment. |
| `--output-dir` | Directory for prepared inputs and results. Use a separate directory for each task, model, and setting. | `runs/single-point` |
| `--model` | Served model ID, required by `run`. | `my-model` |
| `--base-url` | OpenAI-compatible model API address. | `http://127.0.0.1:8000/v1` |

Keep question selection, context, sampling, subtitle, and output settings
consistent between `prepare` and `run`.

## Results

Outputs are saved under `--output-dir`: `inputs/` contains selected QA,
`images/` contains frames, and `results/` contains evaluation results.
Check `frames_summary.json` after preparation and `summary.json` after inference.
Missing, skipped, randomly guessed, or truncated answers cause a nonzero exit
status; answer accuracy is reported separately in the evaluation results.

Run `dramavqa prepare --help` or `dramavqa run --help` for all options.
