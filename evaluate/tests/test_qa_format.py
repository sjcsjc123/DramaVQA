import json
from pathlib import Path

import pytest

try:
    from SinglePointEvalOpen.conan_qa_adapter import convert as convert_for_conan
    from SinglePointEvalOpen.qa_format import load_qa, validate_qa
except ModuleNotFoundError:  # supports running pytest from inside this directory
    from conan_qa_adapter import convert as convert_for_conan
    from qa_format import load_qa, validate_qa


FIXTURE = Path(__file__).parent / "fixtures/latest_qa.json"


def test_latest_format_expands_each_question():
    records, summary = load_qa(FIXTURE)
    assert summary == {"clips": 2, "questions": 3, "levels": ["L2", "L3", "L4", "L5"]}
    assert [r.question_id for r in records] == ["42:0", "42:1", "99:0"]
    assert records[0].time_for("L3").startswith("S1#0001-")


def test_latest_format_rejects_missing_question_list():
    bad = json.loads(FIXTURE.read_text())
    bad["42"]["Q&A"] = {"0": bad["42"]["Q&A"][0]}
    with pytest.raises(ValueError, match="Q&A"):
        validate_qa(bad)


def test_empty_question_list_is_valid_clip():
    data = json.loads(FIXTURE.read_text())
    data["42"]["Q&A"] = []
    assert validate_qa(data)["clips"] == 2
    assert load_qa(FIXTURE)[1]["questions"] == 3


def test_conan_adapter_keeps_all_question_content_and_windows():
    original = json.loads(FIXTURE.read_text())
    adapted = convert_for_conan(original)["single-point"]
    assert list(adapted) == list(original)
    for clip_id, clip in original.items():
        assert {key: value for key, value in adapted[clip_id].items() if key != "Q&A"} == {
            key: value for key, value in clip.items() if key != "Q&A"
        }
        assert list(adapted[clip_id]["Q&A"].values()) == clip["Q&A"]
