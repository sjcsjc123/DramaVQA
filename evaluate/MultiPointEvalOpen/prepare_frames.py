"""Generate the shared L3 scene frames used by multi-point and multi-hop QA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from .qa_format import load_multi_point
from MultiHopEvalOpen.qa_format import load_multi_hop


def frame_qa(questions: list[dict], *, limit_questions: int = 0) -> dict[str, dict]:
    """Build one flat QA clip per distinct L3 window for the existing sampler."""

    windows: dict[str, dict] = {}
    for question in questions[:limit_questions or None]:
        for time in question["src_questions_info"]["Time"]:
            window = time["L3_time"]
            if window in windows:
                continue
            clip = {
                "L2_time": time.get("L2_time", window),
                "L2_frame": time.get("L2_frame", time.get("L3_frame", "not_used")),
                "L3_time": window,
                "L3_frame": time.get("L3_frame", "not_used"),
                "L4_time": window,
                "L4_frame": time.get("L3_frame", "not_used"),
                "L5_time": window,
                "L5_frame": time.get("L3_frame", "not_used"),
                "Q&A": [{"Question": "Frame preparation only",
                         "Options": "A. Prepared\nB. Missing", "Answer": "A",
                         "Dimension": "Frame preparation"}],
            }
            windows[window] = clip
    return {hashlib.sha256(window.encode()).hexdigest(): clip
            for window, clip in windows.items()}


def command(args: argparse.Namespace, qa_path: Path) -> list[str]:
    cmd = [
        str(args.python), "-m", "SinglePointEvalOpen.legacy_compatible_runner",
        "--drama", args.drama, "--question_path", str(qa_path),
        "--video_path", str(args.video_root),
        "--image_path", str(args.output_root / "images"),
        "--sample_frames", str(args.sample_frames),
        "--target_width", str(args.target_width),
        "--level", "L3", "--model_name", "frame-preparation",
        "--serving_mode", "local", "--only_gen_img",
        "--no_render_subtitle", "--subtitle_text", "--use_render_info",
    ]
    if args.use_rendered_video:
        cmd.append("--use_rendered_video")
    return cmd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="one JSON or DramaVQA.zip")
    parser.add_argument("--kind", choices=("multi_point", "multi_hop"), required=True)
    parser.add_argument("--drama", required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-frames", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=392)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--use-rendered-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="write frame QA without extracting")
    args = parser.parse_args(argv)
    if args.sample_frames < 1 or args.target_width < 1 or args.limit_questions < 0:
        parser.error("sample-frames and target-width must be positive; limit nonnegative")
    loader = load_multi_point if args.kind == "multi_point" else load_multi_hop
    adapted, summary = loader(args.qa, drama=args.drama)
    questions = [item for group in adapted.values() for item in group.values()]
    selected = questions[:args.limit_questions or None]
    clips = frame_qa(questions, limit_questions=args.limit_questions)
    if not clips:
        parser.error("no L3 windows in QA")
    qa_path = args.output_root / "inputs" / "frame_qa" / args.kind / f"{args.drama}.json"
    image_root = (args.output_root / "images" / "L3" / str(args.sample_frames)
                  / f"results.w{args.target_width}_nosub" / "img_data")
    print(json.dumps({"source_questions": summary["questions"],
                      "selected_questions": len(selected), "distinct_L3_windows": len(clips),
                      "frame_qa": str(qa_path), "evaluation_image_root": str(image_root)},
                     ensure_ascii=False), flush=True)
    cmd = command(args, qa_path)
    print("command=" + shlex.join(cmd), flush=True)
    if args.dry_run:
        return 0
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(clips, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
