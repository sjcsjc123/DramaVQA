# DramaVQA

**DramaVQA: Probing Detailed Understanding of Ultra-Long TV Dramas**

[Project](https://github.com/xinyuehuo/DramaVQA) | [Videos & Metadata](https://huggingface.co/datasets/198808xc/drama_bench_data) | [Question Data](DramaVQA/) | [Evaluation](evaluate/) | [License](LICENSE)

DramaVQA is a video question answering benchmark for evaluating detailed understanding of TV dramas over extended temporal contexts. It challenges multimodal models to locate visual evidence, recognize characters and events, and reason over narratives that span many episodes.

## Overview

The full single-point evaluation set contains **20,000 multiple-choice questions across 12 TV dramas**, covering approximately **466 hours** of video. Each question has one correct answer among 2–8 options. Questions are checked for uniqueness within the entire drama, allowing the same question to be evaluated with progressively longer video inputs. The project also provides a **2,000-question single-point Mini subset**, multi-point questions, multi-hop questions, and a training set covering **three additional TV dramas**.

Key features include:

- **Four temporal tracks:** evaluate video understanding from minutes to tens of hours.
- **Detailed question coverage:** seven major categories and 28 subcategories cover visual details, actions, events, and reasoning.
- **LLM-assisted annotation with human verification:** questions undergo generation, filtering, human checking and editing, language-prior removal, and global uniqueness verification.
- **Long-context research:** support the study of video retrieval, persistent memory, and reasoning across distant events.

### Evaluation tracks

| Track | Average input duration |
| --- | --- |
| Medium | 7.6 minutes |
| Long | 43.1 minutes |
| Extra-Long | 9.3 hours |
| Ultra-Long (full drama) | 38.9 hours |

The tracks vary the amount of video context available for answering a question; they do not represent four separate collections of 20,000 questions. The released evaluator selects the time windows stored in the question annotations using `--level L2`, `L3`, `L4`, or `L5`. Use these annotation keys when running the CLI; `L6` is not an accepted CLI value.

### Question categories

| Category | Examples of evaluated abilities |
| --- | --- |
| Visual Perception | Character appearance, objects, scenes, and on-screen text |
| Spatial Understanding | Spatial locations, relative positions, and viewpoints |
| Action and Interaction | Body movements, action participants, and object motion |
| Event Understanding | Event summaries, background, outcomes, and motivations |
| Temporal Understanding | Changes, transitions, ordering, and temporal comparisons |
| Relation & Logical Reasoning | Character relations, human-object relations, and causal inference |
| Counting | Object counts and quantity comparisons |

## Data

The dataset is organized into two parts:

- **Videos and metadata:** source videos and supporting metadata are hosted at **[198808xc/drama_bench_data](https://huggingface.co/datasets/198808xc/drama_bench_data)** on Hugging Face.
- **Question data:** question annotations are stored in the **[`DramaVQA/`](DramaVQA/) subdirectory of this repository**.

The broader data collection described in the paper contains 15 TV series totaling approximately 606 hours; 12 series form the evaluation benchmark, and three additional series are reserved for training.

### Videos and metadata

The evaluation media layout uses one directory per drama, with the following contents:

| Path within a drama | Contents |
| --- | --- |
| `video_data/` | Episode videos in MP4 format |
| `face_video_data/` | Videos with face annotations |
| `face_data/` | Face metadata in JSON format |
| `subtitle_data/` | Subtitle metadata in JSON format |
| `timestamp_data/` | Timestamp metadata in CSV format |
| `SHA256SUMS` | File checksums |

English-language dramas use a single directory such as `breaking_bad/`, not
`breaking_bad_1/` or `breaking_bad_2/`. Episode numbering continues across seasons:
if one season ends at `0008`, the next begins at `0009`. QA timestamps, videos,
and all episode metadata use those same continuous numbers. Media merging is
performed separately; the evaluator does not rename files or infer season offsets.

### Question data

All question-data paths below are relative to the project root:

| Collection | Directory | Contents |
| --- | --- | --- |
| Single-point Full | [`DramaVQA/Bench_Data/`](DramaVQA/Bench_Data/) | Full single-point evaluation set with **20,000 questions** |
| Single-point Mini | [`DramaVQA/Bench_Data_mini_2000/`](DramaVQA/Bench_Data_mini_2000/) | Single-point Mini subset with **2,000 questions** |
| Multi-point | [`DramaVQA/Multi-Point/`](DramaVQA/Multi-Point/) | Multi-point questions |
| Multi-hop | [`DramaVQA/Multi-Hop/`](DramaVQA/Multi-Hop/) | Multi-hop questions |
| Training | [`DramaVQA/Training_Set/`](DramaVQA/Training_Set/) | Training questions from **three additional dramas**, separate from the 12 evaluation dramas |

The training set contains `downton_abbey.json` (*Downton Abbey*), `a_lifelong_journey.json` (*A Lifelong Journey*), and `empresses_in_the_palace.json` (*Empresses in the Palace*).

Use the question annotations from this repository together with the corresponding videos and metadata from Hugging Face for evaluation or training.

### Download videos and metadata

Question annotations and evaluation code are included when you clone this repository:

```bash
git clone https://github.com/xinyuehuo/DramaVQA.git
cd DramaVQA
```

Install the Hugging Face CLI, then download the videos and metadata for the drama you want to evaluate. The commands below run from the repository root and use *The Knockout* (`the_knockout`) throughout:

```bash
python -m pip install --upgrade huggingface_hub
hf download 198808xc/drama_bench_data --repo-type dataset \
  --include "the_knockout/*" --local-dir data/drama_bench_data
```

Replace the include pattern for another drama. To download the complete media repository, omit `--include`. Video downloads can be large; download every episode required by the selected context windows, including intermediate episodes for contexts that span multiple episodes.

The evaluator expects a media root containing drama directories with versioned media and metadata:

```text
data/drama_bench_data/<drama>/
├── video_data/
│   ├── version.json
│   └── <active-version>/0001.mp4
├── face_data/
│   ├── version.json
│   └── <active-version>/0001.json
└── subtitle_data/
    ├── version.json
    └── <active-version>/0001.json
```

`--video-root` points to the parent of the drama directories. Face and subtitle metadata support the benchmark's annotated visual inputs and text subtitles. See [source data and rendering](evaluate/README.md#source-data-and-rendering) for version selection, continuous episode numbering, and pre-rendered videos.

## Evaluation

The [`evaluate/`](evaluate/) directory contains the runnable evaluation package. Its public `dramavqa` CLI provides three stages: **validate question annotations**, **prepare model inputs**, and **run evaluation**.

### Supported tasks

| Task | CLI argument | Question input | Context |
| --- | --- | --- | --- |
| Single-point | `--task single-point` | `DramaVQA/Bench_Data/` or `DramaVQA/Bench_Data_mini_2000/` | `L2`–`L5` |
| Multi-point | `--task multi-point` | `DramaVQA/Multi-Point/` | `L3` |
| Multi-hop | `--task multi-hop` | `DramaVQA/Multi-Hop/` | `L3` |
| Paired source questions | `--task source-q1` or `--task source-q2` | The same `DramaVQA/Multi-Hop/` JSON | The same two `L3` scenes |
| Single-point ablations | `--task ablation --mode MODE` | Single-point JSON | `L2`–`L5` |

Single-point evaluation also accepts the training JSON format. Multi-point and multi-hop evaluation use text subtitles and preserve chronological scene ordering. Paired-source modes evaluate each source question within its pair; their results are per pair, rather than a deduplicated source-question metric.

### Installation

The evaluation client requires **Linux, Python 3.10+, FFmpeg/ffprobe**, and access to a compatible model service. From the repository root:

```bash
python -m pip install ./evaluate
python -m dramavqa --help
```

`dramavqa` and `python -m dramavqa` are equivalent. The default backend calls an OpenAI-compatible model API. Set `OPENAI_API_KEY` in your environment if the service requires authentication; use `OPENAI_BASE_URL` or `--base-url` for its endpoint. Set `EVAL_FONT_PATH` to a Chinese-capable font for rendering Chinese labels and subtitles.

### Quick start: single-point evaluation

The following Bash commands run from the repository root. They select ten questions from the Mini subset and sample eight frames per context window for a quick check. Replace the model ID and endpoint with your own service.

```bash
# 1. Validate question annotations without calling a model.
python -m dramavqa validate --task single-point \
  --qa DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --drama the_knockout --limit 10

# 2. Prepare frames and subtitle inputs from the downloaded media.
python -m dramavqa prepare --task single-point \
  --qa DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --drama the_knockout --limit 10 --level L2 --sample-frames 8 \
  --video-root data/drama_bench_data \
  --output-dir runs/single-point/the_knockout

# 3. Evaluate the prepared inputs.
python -m dramavqa run --task single-point \
  --qa DramaVQA/Bench_Data_mini_2000/the_knockout.json \
  --drama the_knockout --limit 10 --level L2 --sample-frames 8 \
  --output-dir runs/single-point/the_knockout \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

Use identical QA input, question limit, level, frame count, and subtitle settings for `prepare` and `run`. `--limit` counts questions, not clips; `--limit 0` selects all questions. Selection follows file order. Eight frames are for quick checks; explicitly configure the frame count and context required by your experiment for benchmark results.

### Automatic question-path selection

Inside this checkout, `--qa` is optional. The CLI discovers the tracked `DramaVQA/`
question directory and selects the file for `--task`, `--drama` and `--subset`:

```bash
python -m dramavqa validate --task single-point --drama the_knockout --limit 10
python -m dramavqa validate --task single-point --subset Bench_Data --drama the_knockout --limit 10
python -m dramavqa validate --task multi-point --drama the_knockout --limit 10
python -m dramavqa validate --task multi-hop --drama the_knockout --limit 10
python -m dramavqa validate --task single-point --subset Training_Set --drama downton_abbey --limit 10
```

The default single-point/ablation subset is `Bench_Data_mini_2000`.
Use `--data-root /path/to/DramaVQA` when running outside the checkout or using
another question-data directory. An explicit `--qa FILE` overrides this lookup.
These options locate question annotations only; `--video-root` still points to
the separately downloaded videos and metadata. Use the exact JSON filename stem
for `--drama`, for example `qin_empire_2`.

### Multi-point and multi-hop evaluation

Use the same three-stage workflow with the corresponding task, QA path, and `--level L3`. For example:

```bash
python -m dramavqa validate --task multi-hop \
  --qa DramaVQA/Multi-Hop/the_knockout.json --drama the_knockout --limit 10

python -m dramavqa prepare --task multi-hop \
  --qa DramaVQA/Multi-Hop/the_knockout.json \
  --drama the_knockout --limit 10 --level L3 --sample-frames 8 \
  --video-root data/drama_bench_data \
  --output-dir runs/multi-hop/the_knockout

python -m dramavqa run --task multi-hop \
  --qa DramaVQA/Multi-Hop/the_knockout.json \
  --drama the_knockout --limit 10 --level L3 --sample-frames 8 \
  --output-dir runs/multi-hop/the_knockout \
  --model my-model --base-url http://127.0.0.1:8000/v1
```

For multi-point evaluation, replace `multi-hop` with `multi-point`, use `DramaVQA/Multi-Point/the_knockout.json`, and choose a separate output directory.

### Results and additional options

Each run directory contains selected annotations under `inputs/`, generated frames under `images/`, and evaluator outputs under `results/`. Input and run manifests record the configuration and prevent reuse with different inputs or model settings. Compatible runs can resume in the same directory.

`frames_summary.json` reports preparation completeness. After evaluation, `summary.json` reports expected, recorded, valid, matched, guessed, and truncated answers, together with a `complete` flag. Missing, skipped, randomly guessed, or truncated answers cause the CLI to exit nonzero. A completed but incorrect model answer still counts as completed inference; consult the evaluator results for accuracy and timing.

The package also provides:

- **Six ablations:** burned subtitles, text subtitles, face galleries, speaker labels, no subtitles, and language-only inputs. See [ablation modes](evaluate/README.md#ablations).
- **Prompt reasoning options:** `--thinking on` or `off` for API single-point and ablation evaluation. See [prompt and thinking settings](evaluate/README.md#full-prompt-and-thinking).
- **An optional Conan backend:** single-point evaluation through an external compatible Conan-Eval checkout and model checkpoint. See [Conan setup](evaluate/README.md#conan-backend).

For full instructions, see the [evaluation README](evaluate/README.md), [advanced commands](evaluate/docs/ADVANCED.md), and [comparison with historical evaluators](evaluate/COMPARISON.md).

## Repository Structure

```text
DramaVQA/
├── README.md
├── LICENSE
├── DramaVQA/                     # Question annotations
│   ├── Bench_Data/               # Full single-point benchmark
│   ├── Bench_Data_mini_2000/     # Single-point Mini subset
│   ├── Multi-Point/              # Multi-point questions
│   ├── Multi-Hop/                # Multi-hop questions
│   └── Training_Set/             # Training questions
└── evaluate/                     # Installable evaluation package
    ├── README.md
    ├── pyproject.toml
    ├── dramavqa/                 # Unified CLI and completion checks
    ├── SinglePointEvalOpen/
    ├── MultiPointEvalOpen/
    ├── MultiHopEvalOpen/
    ├── AblationEvalOpen/
    ├── docs/
    └── tests/
```

The download and evaluation commands create local `data/` and `runs/` directories. Keep downloaded media, model checkpoints, credentials, and generated results out of commits.

## Citation

If you use DramaVQA in your research, please cite:

```bibtex
@misc{huo2026dramavqa,
  title  = {DramaVQA: Probing Detailed Understanding of Ultra-Long TV Dramas},
  author = {Xinyue Huo and Pengfei Chen and Jiannan Ge and Kaiwen Duan and Yunning Cao and Jiacheng Shao and Lingxi Xie and Qi Tian},
  year   = {2026},
  url    = {https://github.com/xinyuehuo/DramaVQA}
}
```

## License

The code and documentation in this repository are released under the **[MIT License](LICENSE)**.
