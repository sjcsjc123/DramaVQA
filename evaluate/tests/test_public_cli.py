"""Public workflow regressions: sampling, validation, resume and completeness."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dramavqa.cli import main, select_questions
from dramavqa.verify import verify_frames, verify_results

FIXTURE = Path(__file__).parent / "fixtures/latest_qa.json"


def test_question_limit_splits_clip_without_mutating_source():
    data = {"a": {"Q&A": [{"id": 1}, {"id": 2}]}, "b": {"Q&A": [{"id": 3}, {"id": 4}]}}
    selected = select_questions(data, 3)
    assert [q["id"] for c in selected.values() for q in c["Q&A"]] == [1, 2, 3]
    assert len(data["b"]["Q&A"]) == 2
    assert select_questions(data, 0) == data
    with pytest.raises(ValueError):
        select_questions(data, -1)


def test_validate_all_input_before_limiting(tmp_path):
    data = json.loads(FIXTURE.read_text())
    bad = dict(next(iter(data.values())))
    bad["Q&A"] = [{"Question": "invalid"}]
    data["bad"] = bad
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(data))
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--task", "single-point", "--qa", str(qa), "--drama", "sample", "--limit", "1"])
    assert exc.value.code == 2


def test_frames_missing_is_failure(tmp_path):
    data = json.loads(FIXTURE.read_text())
    assert not verify_frames(data, tmp_path, "L2", 8)["complete"]
    assert verify_frames(data, tmp_path, "L2", 8, language_only=True)["complete"]


def test_result_presence_does_not_hide_skips_or_guesses(tmp_path):
    p = tmp_path / "q_a_dict.json"
    rows = {"0": {"Q&A": {"Question": "q", "Answer": "A", "result": "B"}}}
    p.write_text(json.dumps(rows))
    assert verify_results(tmp_path, 1)["complete"]  # Wrong answers are still completed inference.
    assert not verify_results(tmp_path, 2)["complete"]
    rows["0"]["Q&A"]["guessed"] = True
    p.write_text(json.dumps(rows))
    assert not verify_results(tmp_path, 1)["complete"]
    rows["0"]["Q&A"].update(guessed=False, skip_reason="answer_extract_failed")
    p.write_text(json.dumps(rows))
    assert not verify_results(tmp_path, 1)["complete"]


def test_multi_scene_results_deduplicate_retries(tmp_path):
    (tmp_path / "results.json").write_text('{"total": 1}')
    (tmp_path / "raw_responses.jsonl").write_text('\n'.join(json.dumps(r) for r in [
        {"qa_type": "B_order", "qa_id": "a", "pred": None, "skip_reason": "api_error"},
        {"qa_type": "B_order", "qa_id": "a", "pred": "B", "skip_reason": None},
    ]))
    assert verify_results(tmp_path, 1)["complete"]


@pytest.mark.parametrize("multi_scene", [False, True])
@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_extracted_letter_does_not_hide_truncation(tmp_path, multi_scene, retry_succeeds):
    if multi_scene:
        (tmp_path / "results.json").write_text('{"total": 1}')
        row = {"qa_type": "B_order", "qa_id": "a", "pred": "B"}
    else:
        (tmp_path / "q_a_dict.json").write_text(json.dumps(
            {"0": {"Q&A": {"Question": "q", "Answer": "B", "result": "B"}}}))
        row = {"key": "0", "extract_ok": True}
    attempts = [{**row, "finish_reason": "length"}]
    if retry_succeeds:
        attempts.append({**row, "finish_reason": "stop"})
    (tmp_path / "raw_responses.jsonl").write_text(
        '\n'.join(json.dumps(attempt) for attempt in attempts))
    report = verify_results(tmp_path, 1)
    assert report["valid"] == 1
    assert report["truncated"] == int(not retry_succeeds)
    assert report["complete"] == retry_succeeds


def test_successful_backend_exit_without_answers_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("dramavqa.cli.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0))
    args = ["run", "--task", "single-point", "--qa", str(FIXTURE), "--drama", "sample",
            "--output-dir", str(tmp_path), "--model", "fixture", "--limit", "1"]
    assert main(args) == 1
    assert not json.loads((tmp_path / "summary.json").read_text())["complete"]
    with pytest.raises(SystemExit) as exc:
        main(args[:-1] + ["2"])
    assert exc.value.code == 2


def test_model_change_requires_separate_results(tmp_path, monkeypatch):
    monkeypatch.setattr("dramavqa.cli.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0))
    args = ["run", "--task", "single-point", "--qa", str(FIXTURE), "--drama", "sample",
            "--output-dir", str(tmp_path), "--model", "model-a"]
    main(args)
    with pytest.raises(SystemExit) as exc:
        main(args[:-1] + ["model-b"])
    assert exc.value.code == 2


def test_all_shipped_python_sources_compile():
    root = Path(__file__).parents[1]
    for package in ("dramavqa", "SinglePointEvalOpen", "MultiPointEvalOpen", "MultiHopEvalOpen", "AblationEvalOpen"):
        for path in (root / package).glob("*.py"):
            compile(path.read_text(), str(path), "exec")


def test_seasonal_frame_directory_names(tmp_path):
    window = "S1#0001-00:00:00.000_S1#0001-00:00:01.000"
    render = tmp_path / window.replace(":", "-").replace("#", "-") / "render"
    render.mkdir(parents=True)
    (render / "frame.jpg").write_bytes(b"fixture")
    assert verify_frames({"0": {"L2_time": window, "Q&A": [1]}}, tmp_path, "L2", 1)["complete"]


def test_public_image_directory_wins_over_legacy_alias(tmp_path):
    from MultiPointEvalOpen.legacy_evaluator import resolve_drama_image_dir
    drama = "da_qin_di_guo_zhi_zong_heng"
    assert resolve_drama_image_dir(str(tmp_path), drama) == "da_qin_di_guo"
    (tmp_path / drama).mkdir()
    assert resolve_drama_image_dir(str(tmp_path), drama) == drama


def test_thinking_is_explicit_and_prevents_mixed_resume(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((kwargs["env"]["SYS_PROMPT_MODE"], kwargs["env"]["THINK_MODE"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("dramavqa.cli.subprocess.run", run)
    monkeypatch.setenv("SYS_PROMPT_MODE", "strict")
    args = ["run", "--task", "single-point", "--qa", str(FIXTURE), "--drama", "sample",
            "--output-dir", str(tmp_path), "--model", "fixture", "--thinking", "on"]
    main(args)
    assert calls == [("full", "1")]
    metadata = json.loads((tmp_path / "run_manifest.json").read_text())
    assert metadata["prompt_mode"] == "full" and metadata["think_mode"] == "1"
    with pytest.raises(SystemExit):
        main(args[:-1] + ["off"])
    monkeypatch.setenv("ENABLE_THINKING", "off")
    with pytest.raises(SystemExit):
        main(args)


@pytest.mark.parametrize("mode", ["none", "no-example", "no_example", "strict"])
def test_removed_prompt_modes_are_rejected(mode):
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--task", "single-point", "--qa", str(FIXTURE),
              "--drama", "sample", "--prompt-mode", mode])
    assert exc.value.code == 2


@pytest.mark.parametrize("module_name", ["SinglePointEvalOpen.legacy_compatible_runner", "AblationEvalOpen.ask_vllm"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("group", [False, True])
def test_full_prompt_variants(module_name, thinking, group, monkeypatch):
    import importlib
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "THINK_MODE", thinking)
    if module_name.startswith("Ablation"):
        monkeypatch.setattr(module, "ABLATION_MODE", "exp2_nosub")
    content = module.GROUP_MESSAGE_SYSTEM if group else module.MESSAGE_SYSTEM
    expected = module._think_prompt() if thinking else (
        (module._group_message_system_head() if group else module._message_system_head())
        + module.MESSAGE_SYSTEM_FORMAT
        + (module.GROUP_MESSAGE_SYSTEM_EXAMPLE if group else module.MESSAGE_SYSTEM_EXAMPLE))
    assert module.resolve_system_content(content) == expected
    assert module.SYS_PROMPT_MODE == "full"
    assert not hasattr(module, "_strict_prompt")


@pytest.mark.parametrize("thinking", [False, True])
def test_full_language_only_prompt(thinking, monkeypatch):
    from AblationEvalOpen import ask_vllm as module
    monkeypatch.setattr(module, "THINK_MODE", thinking)
    monkeypatch.setattr(module, "ABLATION_MODE", "exp6_langonly")
    expected = module._langonly_think_prompt() if thinking else (
        module._langonly_message_system_head() + module.MESSAGE_SYSTEM_FORMAT + module.MESSAGE_SYSTEM_EXAMPLE)
    assert module.resolve_system_content(module.MESSAGE_SYSTEM) == expected


@pytest.mark.parametrize("module_name", ["SinglePointEvalOpen.legacy_compatible_runner", "AblationEvalOpen.ask_vllm"])
@pytest.mark.parametrize("usage", [None, SimpleNamespace(total_tokens=None), SimpleNamespace(total_tokens=12)])
def test_optional_usage_does_not_drop_valid_answer(module_name, usage, capsys):
    import importlib
    module = importlib.import_module(module_name)
    engine = module.LLMInference.__new__(module.LLMInference)
    engine.model_name = engine.short_model_name = "fixture"
    engine.dump_raw_response = lambda *args, **kwargs: None
    engine.openai_inference = lambda msg: SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="<A>"), finish_reason="stop")], usage=usage)
    function = getattr(module.LLMInference.get_llm_response, "__wrapped__", module.LLMInference.get_llm_response)
    answer, tokens, raw, skip = function(engine, [], "0")
    assert (answer, tokens, raw, skip) == ("A", getattr(usage, "total_tokens", None), "<A>", None)
    engine.evaluation({"0": {"Q&A": {"Answer": "A", "result": answer, "total_tokens": tokens}}})
    output = capsys.readouterr().out
    assert "accuracy: 100.00" in output
    assert ("unavailable" in output) == (tokens is None)
