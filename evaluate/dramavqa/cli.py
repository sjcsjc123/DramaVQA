"""Validate inputs, prepare frames, and evaluate through one public interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from zipfile import BadZipFile, ZipFile

from SinglePointEvalOpen.qa_format import validate_qa
from MultiPointEvalOpen.qa_format import validate_multi_point
from MultiHopEvalOpen.qa_format import validate_multi_hop
from . import __version__

TASKS = ("single-point", "multi-point", "multi-hop")


def select_questions(data, limit):
    """Select at most limit questions, retaining source order and clip IDs."""
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if isinstance(data, list):
        return data[:limit or None]
    selected, remaining = {}, limit
    for key, clip in data.items():
        questions = clip["Q&A"][:remaining] if limit else clip["Q&A"]
        if questions:
            selected[key] = {**clip, "Q&A": questions}
            remaining -= len(questions)
        if limit and remaining == 0:
            break
    return selected


def resolve_data_root(args):
    """Locate the question collection root."""
    if args.data_root is not None:
        root = args.data_root.expanduser().resolve()
    else:
        cwd = Path.cwd().resolve()
        candidates = [parent / "DramaVQA" for parent in (cwd, *cwd.parents)]
        candidates.append(Path(__file__).resolve().parents[2] / "DramaVQA")
        root = next((candidate for candidate in candidates
                     if (candidate / "Bench_Data_mini_2000").is_dir()
                     and (candidate / "Multi-Hop").is_dir()), None)
        if root is None:
            raise ValueError("Cannot locate the repository's DramaVQA/ directory. "
                             "Pass --data-root /path/to/DramaVQA or --qa FILE.")
    return root


def resolve_qa_path(args, group):
    """Resolve an explicit input or the checkout's DramaVQA question directory."""
    if args.qa is not None:
        return args.qa
    root = resolve_data_root(args)
    path = root / group / f"{args.drama}.json"
    if not path.is_file():
        raise ValueError(f"Question file not found: {path}. Use the exact JSON filename "
                         "stem for --drama, select the correct --subset, or pass --qa FILE.")
    return path


def load_input(args):
    group = {"multi-point": "Multi-Point", "multi-hop": "Multi-Hop"}.get(args.task, args.subset)
    qa_path = resolve_qa_path(args, group)
    if qa_path.suffix.lower() == ".zip":
        with ZipFile(qa_path) as archive:
            candidates = [n for n in archive.namelist()
                          if n == f"{group}/{args.drama}.json"
                          or n.endswith(f"/{group}/{args.drama}.json")]
            if len(candidates) != 1:
                raise ValueError(f"expected one {group}/{args.drama}.json in ZIP")
            data = json.loads(archive.read(candidates[0]))
    else:
        data = json.loads(qa_path.read_text(encoding="utf-8"))
    validator = (validate_qa if args.task == "single-point" else
                 validate_multi_point if args.task == "multi-point" else validate_multi_hop)
    source_summary = validator(data)
    if args.task == "single-point" and getattr(args, "level", None):
        for clip_id, clip in data.items():
            if not clip.get(f"{args.level}_time") or not clip.get(f"{args.level}_frame"):
                raise ValueError(f"clip {clip_id}: missing {args.level}_time/{args.level}_frame")
    selected = select_questions(data, args.limit)
    if not selected:
        raise ValueError("the selected input contains no questions")
    return selected, source_summary, validator(selected)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"dramavqa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "prepare", "run"):
        p = sub.add_parser(name, help={"validate": "validate QA without a model",
            "prepare": "generate model input frames", "run": "evaluate prepared inputs"}[name])
        p.add_argument("--task", choices=TASKS, required=True)
        p.add_argument("--qa", type=Path, help="JSON or dataset ZIP; with --drama all also accepts a QA directory; overrides --data-root")
        p.add_argument("--data-root", type=Path, help="question-data directory containing Bench_Data, Multi-Point, etc.; default: checkout DramaVQA/")
        p.add_argument("--drama", required=True, help="drama name or all for locally downloaded dramas")
        p.add_argument("--subset", default="Bench_Data_mini_2000",
                       choices=("Bench_Data", "Bench_Data_mini_2000", "Training_Set"))
        p.add_argument("--limit", type=int, default=0, help="maximum questions per drama, not clips; 0 means all")
        p.add_argument("--output-dir", type=Path, help="dedicated directory for this task and model")
        p.add_argument("--video-root", type=Path, help="media root containing drama folders; selects local dramas for all")
        p.add_argument("--image-root", type=Path, help="frame output root; default: OUTPUT_DIR/images; all appends a drama subdirectory")
        p.add_argument("--sample-frames", type=int, default=8)
        p.add_argument("--target-width", type=int, default=392)
        p.add_argument("--level", choices=("L3", "L4", "L5", "L6"))
        p.add_argument("--subtitles", choices=("text", "burn", "none"), default="text")
        p.add_argument("--use-rendered-video", action="store_true")
        p.add_argument("--model", help="served model ID")
        p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
        p.add_argument("--num-workers", type=int, default=1)
        p.add_argument("--max-tokens", type=int, help="output token limit; default: 1000 for API, 4000 for Conan")
        p.add_argument("--api-timeout", type=float, default=120)
        p.add_argument("--prompt-mode", choices=("full",), default="full",
                       help="only full is supported; preserves the original prompt")
        p.add_argument("--thinking", choices=("on", "off"), default="off",
                       help="full prompt reasoning instructions for API single-point; independent of model-side ENABLE_THINKING")
        p.add_argument("--backend", choices=("openai", "conan"), default="openai")
        p.add_argument("--conan-eval", type=Path)
        p.add_argument("--model-path", type=Path, help="Conan checkpoint directory")
        p.add_argument("--python", type=Path, help="Conan environment Python")
        p.add_argument("--gpu-ids", default="0", help="Conan GPU IDs")
    return parser


def frame_root(args):
    return args.image_root or args.output_dir / "images"


def image_data_root(args):
    suffix = "_sub" if args.subtitles == "burn" else "_nosub"
    return frame_root(args) / args.level / str(args.sample_frames) / f"results.w{args.target_width}{suffix}" / "img_data"


def engine_command(args, qa):
    cmd = [sys.executable, "-m"]
    common = ["--qa", str(qa), "--drama", args.drama]
    if args.task in ("multi-point", "multi-hop"):
        if args.command == "prepare":
            return cmd + ["MultiPointEvalOpen.prepare_frames"] + common + [
                "--kind", "multi_point" if args.task == "multi-point" else "multi_hop",
                "--video-root", str(args.video_root), "--output-root", str(args.output_dir),
                "--sample-frames", str(args.sample_frames), "--target-width", str(args.target_width),
            ] + (["--use-rendered-video"] if args.use_rendered_video else [])
        module = "MultiPointEvalOpen.run_eval" if args.task == "multi-point" else "MultiHopEvalOpen.run_eval"
        return cmd + [module] + common + [
            "--image-root", str(image_data_root(args)), "--output-dir", str(args.output_dir / "results"),
            "--model", args.model, "--base-url", args.base_url,
            "--num-workers", str(args.num_workers), "--max-tokens", str(args.max_tokens),
            "--api-timeout", str(args.api_timeout),
        ]
    if args.backend == "conan" and args.command == "run":
        return cmd + ["SinglePointEvalOpen.conan_runner"] + common + [
            "--image-root", str(frame_root(args)), "--output-root", str(args.output_dir / "results"),
            "--conan-eval", str(args.conan_eval), "--model-path", str(args.model_path),
            "--python", str(args.python), "--gpu-ids", args.gpu_ids,
            "--level", args.level, "--sample-frames", str(args.sample_frames),
            "--image-subdir", image_data_root(args).parent.name,
            "--drama-image-map", f"{args.drama}={args.drama}",
            "--max-new-tokens", str(args.max_tokens),
        ]
    result = cmd + ["SinglePointEvalOpen.legacy_compatible_runner",
        "--question_path", str(qa), "--drama", args.drama,
        "--image_path", str(frame_root(args)), "--result_path", str(args.output_dir / "results"),
        "--model_name", args.model or "frame-preparation", "--serving_mode", "local",
        "--sample_frames", str(args.sample_frames), "--target_width", str(args.target_width),
        "--level", args.level, "--num_workers", str(args.num_workers), "--use_render_info"]
    if args.command == "prepare":
        result += ["--only_gen_img", "--video_path", str(args.video_root)]
    if args.subtitles != "burn":
        result.append("--no_render_subtitle")
    if args.subtitles == "text":
        result.append("--subtitle_text")
    if args.use_rendered_video:
        result.append("--use_rendered_video")
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_tokens is None:
        args.max_tokens = 4000 if args.backend == "conan" else 1000
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.drama):
        parser.error("drama must contain only letters, digits, underscores or hyphens")
    if args.limit < 0 or min(args.sample_frames, args.target_width, args.num_workers, args.max_tokens, args.api_timeout) <= 0:
        parser.error("limit must be nonnegative; other numeric limits must be positive")
    multi = args.task in ("multi-point", "multi-hop")
    args.level = args.level or "L3"
    if multi and (args.level != "L3" or args.subtitles != "text"):
        parser.error("multi-scene evaluation uses L3 and text subtitles")
    if multi and args.command == "prepare" and args.image_root:
        parser.error("multi-scene preparation writes OUTPUT_DIR/images; omit --image-root")
    if args.thinking == "on" and (multi or args.backend == "conan"):
        parser.error("--thinking on applies only to API single-point evaluation")
    if args.backend == "conan" and args.task != "single-point":
        parser.error("Conan currently supports only single-point")
    if args.command != "validate":
        if not args.output_dir:
            parser.error("--output-dir is required")
        if args.command == "prepare" and not args.video_root:
            parser.error("prepare requires --video-root")
        if args.command == "run" and args.backend == "openai" and not args.model:
            parser.error("run requires --model")
        if args.command == "run" and args.backend == "conan" and not all((args.conan_eval, args.model_path, args.python)):
            parser.error("Conan requires --conan-eval, --model-path and --python")
    if args.drama == "all":
        return run_all(args, parser)
    return run_one(args, parser)


def run_one(args, parser):
    """Execute one drama using the existing evaluator and completeness checks."""
    try:
        data, source, selected = load_input(args)
    except (ValueError, OSError, KeyError, BadZipFile) as exc:
        parser.error(str(exc))
    print(json.dumps({"source": source, "selected": selected}, ensure_ascii=False), flush=True)
    if args.command == "validate":
        return 0
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True).encode()
    fingerprint = {"qa_sha256": hashlib.sha256(encoded).hexdigest(), "task": args.task,
        "drama": args.drama, "level": args.level, "sample_frames": args.sample_frames,
        "target_width": args.target_width, "subtitles": args.subtitles, "mode": None,
        "use_rendered_video": args.use_rendered_video, "questions": selected["questions"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        parser.error("output directory has different inputs/settings; use a new --output-dir")
    manifest.write_text(json.dumps(fingerprint, indent=2))
    qa = args.output_dir / "inputs" / f"{args.drama}.json"
    qa.parent.mkdir(exist_ok=True)
    qa.write_bytes(encoded)
    env = os.environ.copy()
    env.update(OPENAI_BASE_URL=args.base_url, LOCAL_API_TIMEOUT=str(args.api_timeout),
               API_MAX_TOKENS=str(args.max_tokens), SYS_PROMPT_MODE="full", THINK_MODE="1" if args.thinking == "on" else "0")
    # API credentials are read only from the environment and never written to manifests.
    if args.command == "run":
        run_settings = {"backend": args.backend, "model": args.model, "model_path": str(args.model_path),
                        "max_tokens": args.max_tokens, "base_url": args.base_url,
                        "prompt_mode": args.prompt_mode, "think_mode": env["THINK_MODE"],
                        "enable_thinking": env.get("ENABLE_THINKING", "auto")}
        run_manifest = args.output_dir / "run_manifest.json"
        if run_manifest.exists() and json.loads(run_manifest.read_text()) != run_settings:
            parser.error("output directory belongs to a different model/run; use a new --output-dir")
        run_manifest.write_text(json.dumps(run_settings, indent=2))
    completed = subprocess.run(engine_command(args, qa), env=env, check=False)
    if completed.returncode:
        return completed.returncode
    if args.command == "run":
        from .verify import verify_results
        report = verify_results(args.output_dir / "results", selected["questions"])
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return 0 if report["complete"] else 1
    from .verify import verify_frames
    report = verify_frames(data, image_data_root(args) / args.drama, args.level, args.sample_frames)
    (args.output_dir / "frames_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return 0 if report["complete"] else 1


def discover_dramas(args):
    """Match local media dramas to the selected QA collection, in name order."""
    group = {"multi-point": "Multi-Point", "multi-hop": "Multi-Hop"}.get(args.task, args.subset)
    if args.qa is not None and args.qa.suffix.lower() == ".zip":
        with ZipFile(args.qa) as archive:
            members = [Path(name) for name in archive.namelist()
                       if name.endswith('.json') and Path(name).parent.name == group]
        names = [member.stem for member in members]
        if len(names) != len(set(names)):
            raise ValueError(f"ZIP contains duplicate drama files for {group}")
        selected = {name: args.qa for name in names}
    else:
        if args.qa is not None:
            if not args.qa.is_dir():
                raise ValueError("--drama all requires a QA directory or ZIP; a single JSON needs its drama name")
            folder = args.qa / group if (args.qa / group).is_dir() else args.qa
        else:
            folder = resolve_data_root(args) / group
        selected = {path.stem: path for path in sorted(folder.glob('*.json')) if path.is_file()}
    if args.video_root is not None:
        if not args.video_root.is_dir():
            raise ValueError(f"Video root not found: {args.video_root}")
        media_dirs = ('video_data', 'face_video_data', 'video_face_data', 'video_face_rendered')
        local = {path.name for path in args.video_root.iterdir() if path.is_dir()
                 and any((path / directory).is_dir() for directory in media_dirs)}
        selected = {name: path for name, path in selected.items() if name in local}
    elif args.command == "run" and args.output_dir is not None:
        manifest = args.output_dir / "batch_manifest.json"
        if not manifest.is_file():
            raise ValueError("--drama all requires --video-root, or an output directory prepared with --drama all")
        previous = json.loads(manifest.read_text())
        if previous.get("task") != args.task or previous.get("subset") != args.subset:
            raise ValueError("Prepared batch belongs to a different task/subset; use the same settings")
        names = previous.get("dramas")
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names):
            raise ValueError("Invalid prepared drama list; run prepare again")
        missing = set(names) - set(selected)
        if missing:
            raise ValueError(f"Prepared dramas have no matching QA: {sorted(missing)}")
        selected = {name: selected[name] for name in names}
    else:
        raise ValueError("--drama all requires --video-root to select locally downloaded dramas")
    if not selected:
        raise ValueError(f"No local drama directories have matching {group} question files")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name == "all" for name in selected):
        raise ValueError("Question filenames must be valid drama names, excluding the reserved name all")
    return dict(sorted(selected.items()))


def run_all(args, parser):
    """Run sequentially, isolate each drama's outputs, and report every failure."""
    try:
        dramas = discover_dramas(args)
    except (ValueError, OSError, BadZipFile) as exc:
        parser.error(str(exc))
    if args.command == "prepare":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"task": args.task, "subset": args.subset, "dramas": list(dramas),
                    "video_root": str(args.video_root.resolve())}
        (args.output_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2))
    rows = []
    for drama, qa in dramas.items():
        child = argparse.Namespace(**vars(args))
        child.drama, child.qa = drama, qa
        if args.output_dir is not None:
            child.output_dir = args.output_dir / drama
        if args.image_root is not None:
            child.image_root = args.image_root / drama
        print(f"[{args.command}] {drama}", flush=True)
        row = {"drama": drama}
        try:
            code = run_one(child, parser)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            row["error"] = "Drama failed input/settings validation; see the error above"
        except (ValueError, OSError, KeyError) as exc:
            code = 1
            row["error"] = str(exc)
        row.update(exit_code=code, complete=code == 0)
        if child.output_dir is not None:
            row["output_dir"] = str(child.output_dir)
            summary_name = "summary.json" if args.command == "run" else "frames_summary.json"
            summary_path = child.output_dir / summary_name
            if code == 0 and args.command != "validate" and summary_path.is_file():
                row["summary"] = json.loads(summary_path.read_text())
        rows.append(row)
    failed = sum(not row["complete"] for row in rows)
    report = {"task": args.task, "command": args.command, "dramas": len(rows),
              "successful": len(rows) - failed, "failed": failed,
              "complete": failed == 0, "results": rows}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"batch_{args.command}_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 1 if failed else 0
