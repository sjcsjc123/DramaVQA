import argparse
import json
from pathlib import Path

from AblationEvalOpen.qa_adapter import convert_ablation_qa, load_ablation_qa
from AblationEvalOpen.run_eval import MODES, command


FIXTURE = Path(__file__).parent / "fixtures" / "latest_qa.json"


def test_ablation_adapter_keeps_every_question_and_window():
    original = json.loads(FIXTURE.read_text())
    adapted, summary = load_ablation_qa(FIXTURE)
    assert summary["questions"] == 3
    assert list(adapted) == ["low-level"]
    for clip_id, clip in original.items():
        actual = adapted["low-level"][clip_id]
        assert actual["L3_time"] == clip["L3_time"]
        assert list(actual["Q&A"].values()) == clip["Q&A"]
    assert adapted == convert_ablation_qa(original)


def test_all_ablation_modes_have_explicit_input_policy(tmp_path):
    base = dict(python=Path("python"), drama="friends", video_root=tmp_path / "videos",
                image_path=tmp_path / "images", output_root=tmp_path / "out",
                model="served-model", model_path=None, sample_frames=256,
                level="L3", target_width=392, port=8000, limit=1,
                num_workers=1, speaker_root=None, use_render_info=False,
                use_rendered_video=False, only_gen_img=False, preflight_only=False)
    for mode in MODES:
        cmd = command(argparse.Namespace(**base, mode=mode), tmp_path / "friends.json")
        assert cmd[cmd.index("--ablation_mode") + 1] == mode
        assert "--parallel" in cmd
        assert ("--subtitle_text" in cmd) == (mode in {"exp2_nosub", "exp3_gallery", "exp4_speaker"})
        assert ("--no_render_subtitle" in cmd) == (mode != "exp1_burn")


def test_gallery_cache_import_works_as_installed_package(monkeypatch, tmp_path):
    """Gallery loading must not rely on adding a source directory to sys.path."""
    import threading
    from AblationEvalOpen import ask_vllm, gallery_cache

    engine = ask_vllm.LLMInference.__new__(ask_vllm.LLMInference)
    engine._gallery_clip_lock = threading.Lock()
    engine._gallery_clip_cache = {}
    expected = {"gallery": [], "frames": [], "frame_subs": []}
    monkeypatch.setattr(gallery_cache, "cache_path", lambda *args: tmp_path / "cache")
    monkeypatch.setattr(gallery_cache, "identity", lambda *args: {"sample": True})
    monkeypatch.setattr(gallery_cache, "load", lambda *args: expected)
    assert engine._get_gallery_clip("sample-window") is expected


def test_offline_preparation_reads_current_flat_windows(tmp_path):
    from AblationEvalOpen.prepare_local_inputs import times_for
    path = tmp_path / "breaking_bad.json"
    window = "0008-00:00:00.000_0009-00:00:01.000"
    path.write_text(json.dumps({"clip": {"L3_time": window, "Q&A": []}}))
    assert times_for(path, "L3") == [window]


def test_speaker_metadata_uses_continuous_episode_in_drama_directory():
    from AblationEvalOpen.ask_vllm import LLMInference
    engine = LLMInference.__new__(LLMInference)
    engine.drama = "breaking_bad"
    assert engine._speaker_episode("9_123.jpg") == ("breaking_bad", "0009", 123)
