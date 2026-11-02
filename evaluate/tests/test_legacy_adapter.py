from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image

module = pytest.importorskip("SinglePointEvalOpen.legacy_compatible_runner")


FIXTURE = Path(__file__).parent / "fixtures/latest_qa.json"


def test_legacy_runner_adapts_flat_qa_without_losing_questions(tmp_path):
    time_range = "S1#0001-00:00:00.000_S1#0001-00:00:04.000"
    clip = time_range.replace(":", "-").replace("#", "-")
    render = tmp_path / clip / "render"
    render.mkdir(parents=True)
    (render / "0001.jpg").write_bytes(b"fixture")

    engine = module.LLMInference.__new__(module.LLMInference)
    engine.question_path = str(FIXTURE)
    engine.level = "L2"
    engine.sample_frames = 1
    engine.limit = 1
    engine.shard_index = 0
    engine.shard_total = 1
    engine.image_path = str(tmp_path)
    engine.drama = "friends"
    engine.skipped_videos = []
    engine.skipped_clips = []
    engine.level_info = {level: {"frames": [], "times": []} for level in ["L2", "L3", "L4", "L5", "L6"]}
    engine.get_level_time = lambda value: value
    engine.count_jpg_files = lambda path: 1
    engine.read_question_info()

    assert len(engine.questions["L2"]) == 1
    assert sum(len(item["Q&A"]) for item in engine.questions["L2"].values()) == 2
    assert len(engine.build_dict()) == 2


def test_video_reader_keeps_cfr_frame_order(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required for the fallback integration test")
    pytest.importorskip("cv2")
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for index, color in enumerate(colors):
        Image.new("RGB", (64, 64), color).save(tmp_path / f"{index:03d}.png")
    video = tmp_path / "frames.mkv"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-framerate", "2", "-i", str(tmp_path / "%03d.png"),
         "-c:v", "ffv1", str(video)],
        check=True,
    )
    reader = module.VideoReader(str(video), ctx=module.cpu(0), num_threads=1)
    assert len(reader) == 5
    frames = reader.get_batch([0, 2, 4]).asnumpy()
    assert frames.shape == (3, 64, 64, 3)
    for frame, color in zip(frames, (colors[0], colors[2], colors[4])):
        assert all(abs(int(value) - target) < 12 for value, target in zip(frame[32, 32], color))
