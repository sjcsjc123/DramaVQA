# DramaVQA Evaluation

Run the commands below from the repository root. Requires Python 3.10+,
FFmpeg/ffprobe, and a running OpenAI-compatible vision-language model service.

## Install and download data

```bash
python -m pip install ./evaluate huggingface_hub
hf download 198808xc/drama_bench_data --repo-type dataset \
  --include 'the_knockout/*' --local-dir data/drama_bench_data
```

Questions are included in `DramaVQA/`. Download videos and metadata from
[Hugging Face](https://huggingface.co/datasets/198808xc/drama_bench_data).
Replace `the_knockout` to download another drama. The local media layout is:

```text
data/drama_bench_data/<drama>/
├── video_data/0001.mp4
├── face_video_data/0001.mp4
├── face_data/0001.json
└── subtitle_data/0001.json
```

`--video-root` points to the parent of the drama directories. Episode numbers
continue across seasons within one drama directory.

## Evaluate

Run each experiment in three steps:

1. `validate`: check the question format and selected question count; no model is called.
2. `prepare`: extract the frames and subtitle inputs needed by those questions.
3. `run`: send the prepared inputs to your model service and save predictions and scores.

The examples use ten questions and eight frames. Replace `my-model` and the API
address with your model service. Set `OPENAI_API_KEY` if authentication is
required.

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

### Common settings

Keep the drama, question input, limit, level, frame count, and subtitle settings
identical between `prepare` and `run`. Use a separate output directory for each
task, model, and setting. To evaluate all questions, omit `--limit` from all three
steps and set the required `--sample-frames`.

`--drama` matches the QA filename stem and media directory. Add
`--subset Bench_Data` to all three single-point commands for the full benchmark,
or use `--qa /path/to/questions.json` to select an input file explicitly.

The single-point evaluation tracks are L3–L6; the examples above use L3.
Multi-point, paired-source controls, and multi-hop use L3 source-scene windows.

## Results

Outputs are saved under `--output-dir`: `inputs/` contains selected QA,
`images/` contains frames, and `results/` contains evaluation results.
Check `frames_summary.json` after preparation and `summary.json` after inference.
Missing, skipped, randomly guessed, or truncated answers cause a nonzero exit
status; answer accuracy is reported separately in the evaluation results.

Run `dramavqa prepare --help` or `dramavqa run --help` for all options.
