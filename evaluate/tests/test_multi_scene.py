import json
from pathlib import Path

from PIL import Image
import pytest

from MultiPointEvalOpen.qa_format import normalize_multi_point, validate_multi_point
from MultiHopEvalOpen.qa_format import normalize_multi_hop, validate_multi_hop


EARLY = "S1#0001-00:00:01.000_S1#0001-00:00:02.000"
LATE = "S1#0002-00:00:01.000_S1#0002-00:00:02.000"
OPTIONS = "A. Yes\nB. No"


def multi_point_row(dimension="Two-Event Temporal Comparison(TTC-2)"):
    count = 3 if "Three-Event" in dimension else 2
    return {
        "Question": "Which scene is earlier?", "Options": OPTIONS,
        "Answer": "A", "Dimension": dimension,
        "src_questions_info": {
            "Question": [f"Source {i}" for i in range(count)],
            "Answers": ["Yes"] * count,
            "Time": [{"L3_time": LATE if i == 0 else EARLY} for i in range(count)],
        },
    }


def multi_hop_row():
    return {
        "Question": "What follows?", "Options": OPTIONS,
        "Answer": "A", "Dimension": "Temporal Sequence",
        "src_questions_info": {
            "Bridge_Character": "Person A",
            "Question": ["First source?", "Second source?"],
            "Options": [OPTIONS, OPTIONS],
            "Answers": ["A", "B"],
            "Dimension": ["Action", "Object"],
            "Time": [{"L3_time": LATE}, {"L3_time": EARLY}],
        },
    }


def test_multi_point_three_types_and_stable_ids():
    rows = [multi_point_row(), multi_point_row("Three-Event Temporal(TTC-3)"),
            multi_point_row("Cross-Event Character Identity(CECI)")]
    adapted = normalize_multi_point(rows)
    assert {key: len(value) for key, value in adapted.items()} == {
        "B_order": 1, "B_sort": 1, "D_same": 1
    }
    assert next(iter(adapted["B_order"].values()))["L3_time"] == [LATE, EARLY]
    assert set(adapted["B_order"]) == set(normalize_multi_point(list(reversed(rows)))["B_order"])
    assert validate_multi_point(rows)["questions"] == 3


def test_multi_hop_and_paired_sources_keep_same_two_windows():
    row = multi_hop_row()
    ids = []
    for mode, question, answer in (("multi_hop", "What follows?", "A"),
                                   ("source_q1", "First source?", "A"),
                                   ("source_q2", "Second source?", "B")):
        adapted = normalize_multi_hop([row], mode=mode)["multi_hop"]
        ids.extend(adapted)
        item = next(iter(adapted.values()))
        assert (item["Question"], item["Answer"], item["L3_time"]) == (
            question, answer, [LATE, EARLY]
        )
    assert len(set(ids)) == 1  # join results across the three output directories
    assert validate_multi_hop([row])["paired_source_questions"] == 2


def test_multi_scene_validation_rejects_bad_window_or_source_answer():
    bad = multi_point_row()
    bad["src_questions_info"]["Time"][0]["L3_time"] = "bad"
    with pytest.raises(ValueError, match="L3_time"):
        normalize_multi_point([bad])
    bad_hop = multi_hop_row()
    bad_hop["src_questions_info"]["Answers"][1] = "Z"
    with pytest.raises(ValueError, match="absent from Options"):
        normalize_multi_hop([bad_hop], mode="source_q2")


def test_unchanged_multi_scene_evaluator_reads_images_and_sorts_scenes(tmp_path: Path):
    from MultiPointEvalOpen.legacy_evaluator import MultiPointEvaluator, l3_time_to_dir_name

    for window in (LATE, EARLY):
        folder = tmp_path / "images" / "sample_drama" / l3_time_to_dir_name(window) / "render"
        folder.mkdir(parents=True)
        Image.new("RGB", (28, 28), "red").save(folder / "0.jpg")
    data = normalize_multi_hop([multi_hop_row()], mode="source_q2")
    messages_seen = []
    evaluator = MultiPointEvaluator(
        base_url="http://localhost:8000/v1", api_key="EMPTY", model="test-model",
        image_root=str(tmp_path / "images"), output_dir=str(tmp_path / "result"),
        num_workers=1,
    )

    def fake_call(messages, _key):
        messages_seen.append(messages)
        return "<B>", 1, "stop"

    evaluator._call_api = fake_call
    evaluator.run("unused.json", "sample_drama", qa_data=data)
    summary = json.loads((tmp_path / "result" / "results.json").read_text())
    assert (summary["total"], summary["correct"]) == (1, 1)
    labels = [part["text"] for part in messages_seen[0][1]["content"]
              if part.get("type") == "text" and part.get("text", "").startswith("=== Scene")]
    assert labels == ["=== Scene 2 ===", "=== Scene 1 ==="]
