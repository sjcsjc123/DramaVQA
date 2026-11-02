"""Run one historical single-point ablation with the public flat QA files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .qa_adapter import load_ablation_qa


MODES = (
    "exp1_burn", "exp2_nosub", "exp3_gallery", "exp4_speaker",
    "exp5_nosub", "exp6_langonly",
)


def command(args: argparse.Namespace, qa_path: Path) -> list[str]:
    script = Path(__file__).with_name("ask_vllm.py")
    cmd = [
        str(args.python), str(script),
        "--drama", args.drama,
        "--question_path", str(qa_path),
        "--video_path", str(args.video_root),
        "--image_path", str(args.image_path),
        "--result_path", str(args.output_root / args.mode),
        "--model_name", args.model,
        "--sample_frames", str(args.sample_frames),
        "--level", args.level,
        "--target_width", str(args.target_width),
        "--port", str(args.port),
        "--serving_mode", "local",
        "--ablation_mode", args.mode,
        "--parallel",
        "--limit", str(args.limit),
        "--num_workers", str(args.num_workers),
    ]
    if args.model_path:
        cmd.extend(("--model_path", str(args.model_path)))
    if args.speaker_root:
        cmd.extend(("--speaker_root", str(args.speaker_root)))
    if args.use_render_info:
        cmd.append("--use_render_info")
    if args.use_rendered_video:
        cmd.append("--use_rendered_video")
    if args.only_gen_img:
        cmd.append("--only_gen_img")
    if args.preflight_only:
        cmd.append("--preflight_only")
    if args.mode in {"exp2_nosub", "exp3_gallery", "exp4_speaker"}:
        cmd.extend(("--no_render_subtitle", "--subtitle_text"))
    elif args.mode in {"exp5_nosub", "exp6_langonly"}:
        cmd.append("--no_render_subtitle")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="one flat JSON or DramaVQA.zip")
    parser.add_argument("--subset", choices=("Bench_Data", "Bench_Data_mini_2000"),
                        default="Bench_Data_mini_2000", help="ZIP member group")
    parser.add_argument("--drama", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="show command without writing or evaluating")
    parser.add_argument("--prepare-only", action="store_true", help="write adapted QA without evaluating")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--image-path", type=Path, help="historical evaluator's image root")
    parser.add_argument("--model")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--level", choices=("L2", "L3", "L4", "L5"), default="L3")
    parser.add_argument("--sample-frames", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=392)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="number of clips, not questions")
    parser.add_argument("--speaker-root", type=Path)
    parser.add_argument("--use-render-info", action="store_true")
    parser.add_argument("--use-rendered-video", action="store_true")
    parser.add_argument("--only-gen-img", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    adapted, summary = load_ablation_qa(args.qa, drama=args.drama, subset=args.subset)
    print(json.dumps({"drama": args.drama, "mode": args.mode, **summary}, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0
    if not args.output_root or not args.video_root or not args.image_path or not args.model:
        parser.error("--output-root, --video-root, --image-path and --model are required")
    if args.sample_frames < 1 or args.num_workers < 1 or args.limit < 0:
        parser.error("frame/worker counts must be positive and limit nonnegative")
    if args.preflight_only and args.mode != "exp4_speaker":
        parser.error("--preflight-only applies only to exp4_speaker")
    qa_path = args.output_root / "inputs" / "ablation_qa" / f"{args.drama}.json"
    cmd = command(args, qa_path)
    print("command=" + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(adapted, ensure_ascii=False, indent=2), encoding="utf-8")
    print("adapted_qa=" + str(qa_path), flush=True)
    if args.prepare_only:
        return 0
    env = os.environ.copy()
    env.setdefault("VLLM_SERVED_MODEL_NAME", args.model)
    env.setdefault("GALLERY_CACHE_ROOT", str(args.output_root / "gallery_cache"))
    (args.output_root / args.mode).mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
