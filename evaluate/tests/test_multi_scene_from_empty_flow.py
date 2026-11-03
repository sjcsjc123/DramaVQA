"""Exercise public multi-scene QA from source videos through a scored answer."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from MultiPointEvalOpen.qa_format import load_multi_point


@pytest.mark.parametrize("drama,episodes", [("sample_drama", (1, 2)), ("breaking_bad", (8, 9))])
def test_multi_point_video_to_answer(tmp_path: Path, drama, episodes):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for source-video extraction")
    pytest.importorskip("cv2")
    pytest.importorskip("tqdm")
    from MultiPointEvalOpen.legacy_evaluator import MultiPointEvaluator

    video_dir = tmp_path / "videos" / drama / "video_data" / "v1"
    video_dir.mkdir(parents=True)
    (video_dir.parent / "version.json").write_text(
        json.dumps({"video_step": {"active_version": "v1"}}), encoding="utf-8"
    )
    for episode in episodes:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "testsrc=size=64x64:rate=25:duration=2", "-c:v", "ffv1",
             str(video_dir / f"{episode:04}.mkv")], check=True,
        )
    late = f"{episodes[1]:04d}-00:00:00.000_{episodes[1]:04d}-00:00:01.960"
    early = f"{episodes[0]:04d}-00:00:00.000_{episodes[0]:04d}-00:00:01.960"
    qa_path = tmp_path / f"{drama}.json"
    qa_path.write_text(json.dumps([{
        "Question": "Which scene came first?", "Options": "A. Scene 2\nB. Scene 1",
        "Answer": "A", "Dimension": "Two-Event Temporal Comparison(TTC-2)",
        "src_questions_info": {
            "Question": ["Source one", "Source two"], "Answers": ["one", "two"],
            "Time": [{"L3_time": late}, {"L3_time": early}],
        },
    }]), encoding="utf-8")
    output_root = tmp_path / "output"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    subprocess.run(
        [sys.executable, "-m", "MultiPointEvalOpen.prepare_frames",
         "--qa", str(qa_path), "--kind", "multi_point", "--drama", drama,
         "--video-root", str(tmp_path / "videos"), "--output-root", str(output_root),
         "--sample-frames", "4", "--target-width", "112"],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    image_root = output_root / "images" / "L3" / "4" / "results.w112_nosub" / "img_data"
    for window in (late, early):
        render = image_root / drama / window.replace(":", "-") / "render"
        assert len(list(render.glob("*.jpg"))) == 4
    adapted, _ = load_multi_point(qa_path)
    evaluator = MultiPointEvaluator(
        base_url="http://localhost:8000/v1", api_key="EMPTY", model="fixture",
        image_root=str(image_root), output_dir=str(tmp_path / "answers"), num_workers=1,
    )
    seen = []

    def fixed_response(messages, _key):
        seen.append(messages)
        return "<A>", 1, "stop"

    evaluator._call_api = fixed_response
    evaluator.run(str(qa_path), drama, qa_data=adapted)
    result = json.loads((tmp_path / "answers" / "results.json").read_text())
    assert result["total"] == result["correct"] == 1
    assert sum(part.get("type") == "image_url" for part in seen[0][1]["content"]) == 8
