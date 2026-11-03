"""Run the existing Conan evaluator with current flat single-point QA.

This launcher converts only the QA container shape. The Conan model, image
selection, multi-step inference, and answer extraction stay in Conan-Eval.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

if __package__:
    from .conan_qa_adapter import convert
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SinglePointEvalOpen.conan_qa_adapter import convert


def command(args: argparse.Namespace) -> list[str]:
    resolution = args.resolution
    if resolution is None:
        resolution = 0.1 if args.drama == "the_knockout" else 0.2
    return [
        str(args.python), str(args.conan_eval / "main_singlehop.py"),
        "--benchmark", "singlehop", "--mode", "step",
        "--gpu_ids", args.gpu_ids, "--num_processes", str(args.num_processes),
        "--model_path", str(args.model_path), "--model_type", "Qwen2_5_VL",
        "--max_frames", str(args.sample_frames), "--max_steps", str(args.max_steps),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature), "--drama", args.drama,
        "--level", args.level, "--resolution", str(resolution),
        "--question", f"{args.drama}.json",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="current flat QA JSON")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conan-eval", type=Path, required=True, help="directory with main_singlehop.py")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True, help="Python in Conan's environment")
    parser.add_argument("--drama", required=True)
    parser.add_argument("--level", default="L2", choices=("L2", "L3", "L4", "L5", "L6"))
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--image-subdir", default="results.w392_nosub")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--resolution", type=float, default=None)
    parser.add_argument("--drama-image-map", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("qa", "image_root", "output_root", "conan_eval", "model_path", "python"):
        setattr(args, name, getattr(args, name).resolve())
    for path, label in ((args.qa, "QA"), (args.conan_eval / "main_singlehop.py", "Conan entry"),
                        (args.conan_eval / "singlehop.py", "Conan QA loader"),
                        (args.model_path, "model"), (args.python, "Python"),
                        (args.image_root, "images")):
        if not path.exists():
            raise SystemExit(f"{label} does not exist: {path}")
    if args.sample_frames < 1 or args.num_processes < 1 or args.max_steps < 1:
        raise SystemExit("sample-frames, num-processes and max-steps must be positive")
    data = json.loads(args.qa.read_text(encoding="utf-8"))
    converted = convert(data)
    question_count = sum(len(clip["Q&A"]) for clip in converted["single-point"].values())
    if not question_count:
        raise SystemExit("QA contains no questions")
    qa_dir = args.output_root / "inputs" / "conan_qa"
    output_file = args.output_root / args.level / str(args.sample_frames) / "qa_data" / args.drama / "conan" / "q_a_dict.json"
    env = os.environ.copy()
    env.update({
        "CONAN_USE_DRAMA_QA": "1", "CONAN_QUESTIONS_DIR": str(qa_dir.resolve()),
        "CONAN_IMAGE_ROOT": str(args.image_root.resolve()),
        "CONAN_IMAGE_SUBDIR": args.image_subdir,
        "CONAN_OUTPUT_ROOT": str(args.output_root.resolve()),
        "CONAN_DRAMA_IMG_MAP": args.drama_image_map,
        "CONAN_QA_LEVEL": args.level,
    })
    conda_prefix = args.python.parent.parent
    env["PATH"] = f"{conda_prefix / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{conda_prefix / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("CUDA_HOME", str(conda_prefix))
    cmd = command(args)
    print(f"questions={question_count} converted_qa={qa_dir / (args.drama + '.json')}", flush=True)
    print(f"result={output_file}", flush=True)
    print("command=" + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / f"{args.drama}.json").write_text(
        json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return subprocess.run(cmd, cwd=args.conan_eval, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
