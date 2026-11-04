"""Exercise the compatibility sampler from a source video and current QA."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from SinglePointEvalOpen import legacy_compatible_runner as compatibility


@pytest.mark.parametrize("layout", ["versioned", "flat", "rendered"])
@pytest.mark.parametrize("drama,episode", [("the_knockout", 1), ("breaking_bad", 9)])
def test_from_empty_video_to_compatibility_input(tmp_path, drama, episode, layout):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for source-video extraction")
    pytest.importorskip("cv2")
    pytest.importorskip("tqdm")

    media_root = tmp_path / "videos" / drama
    video_dir = media_root / ("face_video_data" if layout == "rendered" else "video_data")
    if layout == "versioned":
        video_dir = video_dir / "v1"
    video_dir.mkdir(parents=True)
    if layout == "versioned":
        (video_dir.parent / "version.json").write_text(
            json.dumps({"video_step": {"active_version": "v1"}}), encoding="utf-8"
        )
    subtitle_dir = media_root / "subtitle_data"
    subtitle_dir.mkdir()
    (subtitle_dir / f"{episode:04d}.json").write_text(json.dumps({
        "object_list": {"0": {"start_frame": 0, "end_frame": 49, "text": "visible dialogue"}}
    }))
    face_dir = media_root / "face_data"
    face_dir.mkdir()
    (face_dir / f"{episode:04d}.json").write_text(json.dumps({"object_list": {
        str(i): {"start_frame": i, "end_frame": i,
                 "id_list": {"0": {"name": "Alice", "score": 0.99}},
                 "geometry": {"active_frame_list": {"0": {"box": {
                     "left": 1, "top": 20, "right": 40, "bottom": 50}}}}}
        for i in range(50)
    }}))
    # No speaker_data/.workspace/new_roles_map.json is provided.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
         "testsrc=size=64x64:rate=25:duration=2", "-c:v", "ffv1",
         str(video_dir / f"{episode:04d}.mkv")],
        check=True,
    )

    window = f"{episode:04d}-00:00:00.000_{episode:04d}-00:00:01.960"
    clip = {f"L{level}_time": window for level in range(2, 6)}
    clip.update({f"L{level}_frame": f"{episode:04d}-00000_{episode:04d}-00049"
                 for level in range(2, 6)})
    clip["Q&A"] = [
        {"Question": f"What is visible in question {index}?",
         "Options": "A. A test pattern\nB. A blank screen", "Answer": "A",
         "Dimension": "Object Presence"}
        for index in range(2)
    ]
    qa_path = tmp_path / "questions.json"
    qa_path.write_text(json.dumps({"1": clip}), encoding="utf-8")
    image_root = tmp_path / f"{drama}_outputs"  # Parent contains the drama name.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    subprocess.run(
        [sys.executable, "-m", "SinglePointEvalOpen.legacy_compatible_runner",
         "--drama", drama, "--question_path", str(qa_path),
         "--video_path", str(tmp_path / "videos"), "--image_path", str(image_root),
         "--sample_frames", "4", "--level", "L2", "--only_gen_img",
         "--no_render_subtitle", "--subtitle_text", "--model_name", "fixture", "--serving_mode", "local"]
        + (["--use_rendered_video"] if layout == "rendered" else []),
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )

    render_dir = (image_root / "L2" / "4" / "results.1" / "img_data" / drama
                  / window.replace(":", "-") / "render")
    assert len(list(render_dir.glob("*.jpg"))) == 4
    subtitles = json.loads((render_dir.parent / "subtitles.json").read_text())
    assert len(subtitles) == 4
    assert set(subtitles.values()) == {"visible dialogue"}

    engine = compatibility.LLMInference.__new__(compatibility.LLMInference)
    engine.question_path = str(qa_path)
    engine.level = "L2"
    engine.sample_frames = 4
    engine.limit = 0
    engine.shard_index = 0
    engine.shard_total = 1
    engine.image_path = str(render_dir.parent.parent)
    engine.drama = drama
    engine.skipped_videos = []
    engine.skipped_clips = []
    engine.level_info = {level: {"frames": [], "times": []} for level in ["L2", "L3", "L4", "L5", "L6"]}
    engine.get_level_time = lambda value: value
    engine.count_jpg_files = lambda path: len(list(Path(path).glob("*.jpg")))
    engine.read_question_info()
    assert len(engine.build_dict()) == 2
    assert engine.skipped_clips == []
