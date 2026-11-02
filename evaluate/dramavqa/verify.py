"""Report missing/skipped answers separately from model accuracy."""

import json
import re
from pathlib import Path


def verify_results(root: Path, expected: int) -> dict:
    files = list(root.rglob("q_a_dict.json")) + list(root.rglob("results.json"))
    answers = []
    def visit(value):
        if isinstance(value, dict):
            if "Q&A" in value and isinstance(value["Q&A"], dict) and "Question" in value["Q&A"]:
                answers.append(value["Q&A"])
            elif "pred" in value or "prediction" in value:
                answers.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    for path in files:
        if path.name == "results.json":
            raw = path.with_name("raw_responses.jsonl")
            latest = {}
            if raw.exists():
                for line in raw.read_text().splitlines():
                    if line.strip():
                        row = json.loads(line)
                        latest[(row.get("qa_type"), row.get("qa_id"))] = row
            answers.extend(latest.values())
        else:
            visit(json.loads(path.read_text()))
    def prediction(answer):
        return answer.get("result", answer.get("pred", answer.get("prediction")))
    def valid_answer(answer):
        value = prediction(answer)
        options = answer.get("Options", answer.get("options", ""))
        letters = set(re.findall(r"^\s*([A-Z])[.、,，):：]", options, re.MULTILINE))
        return isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z]", value)) and (not letters or value.upper() in letters) and not answer.get("skip_reason")
    valid = sum(bool(valid_answer(a)) for a in answers)
    matched = sum(valid_answer(a) and prediction(a) == a.get("Answer", a.get("answer")) for a in answers)
    guessed = sum(bool(a.get("guessed")) for a in answers)
    # A letter extracted from unfinished reasoning is not a completed response.
    # Retries are append-only, so only the last response for each question counts.
    truncated = 0
    for raw in root.rglob("raw_responses.jsonl"):
        latest = {}
        for index, line in enumerate(raw.read_text().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = [row.get(name) for name in ("qa_type", "qa_id", "key")]
            key = json.dumps(identity) if any(value is not None for value in identity) else str(index)
            latest[key] = row
        truncated += sum(row.get("finish_reason") == "length" for row in latest.values())
    return {"expected": expected, "recorded": len(answers), "valid": valid,
            "guessed": guessed, "truncated": truncated, "matched": matched,
            "complete": len(answers) == expected and valid == expected and not guessed and not truncated,
            "result_files": [str(p) for p in files]}


def verify_frames(data, root: Path, level: str, count: int, *, language_only=False):
    """Require every selected window to have its complete set of sampled images."""
    if language_only:
        return {"complete": True, "windows": 0, "reason": "language-only mode"}
    if isinstance(data, list):
        windows = {t["L3_time"] for q in data for t in q["src_questions_info"]["Time"]}
    else:
        windows = {q[f"{level}_time"] for q in data.values() if q["Q&A"]}
    missing = []
    for window in sorted(windows):
        render = root / window.replace(":", "-").replace("#", "-") / "render"
        actual = len(list(render.glob("*.jpg")))
        if actual != count:
            missing.append({"window": window, "expected_frames": count, "actual_frames": actual})
    return {"complete": not missing, "windows": len(windows), "missing": missing}
