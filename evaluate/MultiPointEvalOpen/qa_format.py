"""Validate and adapt the public DramaVQA Multi-Point JSON files.

The adapter changes only the QA container. Scene order, image loading,
inference, answer extraction, and scoring stay in ``legacy_evaluator``.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile


DIMENSION_TO_TYPE = {
    "Two-Event Temporal Comparison(TTC-2)": "B_order",
    "Three-Event Temporal(TTC-3)": "B_sort",
    "Cross-Event Character Identity(CECI)": "D_same",
}
_WINDOW_ENDPOINT = re.compile(r"(?:S\d+#)?\d+-\d{2}:\d{2}:\d{2}\.\d{3}")
_OPTION = re.compile(r"^\s*([A-Z])\s*[.、,，):：]", re.MULTILINE)
_DRAMA = re.compile(r"[A-Za-z0-9_.-]+")


def _text(value: Any, label: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {label} must be nonempty text")
    return value.strip()


def _window(value: Any, where: str) -> str:
    value = _text(value, "L3_time", where)
    parts = value.split("_")
    if len(parts) != 2 or any(_WINDOW_ENDPOINT.fullmatch(part) is None for part in parts):
        raise ValueError(f"{where}: invalid L3_time window {value!r}")
    return value


def validate_multi_point(data: Any, *, source: str = "Multi-Point") -> dict[str, Any]:
    """Check every field needed for the unchanged multi-scene evaluator."""

    if not isinstance(data, list):
        raise ValueError(f"{source}: expected a list of questions")
    dimensions: Counter[str] = Counter()
    for index, row in enumerate(data):
        where = f"{source}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected an object")
        for field in ("Question", "Options", "Dimension"):
            _text(row.get(field), field, where)
        dimension = row["Dimension"]
        if dimension not in DIMENSION_TO_TYPE:
            raise ValueError(f"{where}: unsupported Dimension {dimension!r}")
        answer = _text(row.get("Answer"), "Answer", where)
        letters = {match.group(1) for match in _OPTION.finditer(row["Options"])}
        if len(answer) != 1 or answer not in letters:
            raise ValueError(f"{where}: Answer {answer!r} is absent from Options")
        info = row.get("src_questions_info")
        if not isinstance(info, dict):
            raise ValueError(f"{where}: src_questions_info must be an object")
        expected = 3 if dimension == "Three-Event Temporal(TTC-3)" else 2
        for field in ("Question", "Answers", "Time"):
            values = info.get(field)
            if not isinstance(values, list) or len(values) != expected:
                raise ValueError(f"{where}: src_questions_info.{field} must contain {expected} items")
        for scene, item in enumerate(info["Time"]):
            if not isinstance(item, dict):
                raise ValueError(f"{where}: Time[{scene}] must be an object")
            _window(item.get("L3_time"), f"{where} Time[{scene}]")
        dimensions[dimension] += 1
    return {"questions": len(data), "dimensions": dict(dimensions)}


def normalize_multi_point(data: Any, *, source: str = "Multi-Point") -> dict[str, dict[str, dict[str, Any]]]:
    """Map public rows to the historical ``type -> id -> QA`` container."""

    validate_multi_point(data, source=source)
    output: dict[str, dict[str, dict[str, Any]]] = {
        "B_order": {}, "B_sort": {}, "D_same": {}
    }
    for index, row in enumerate(data):
        qa_type = DIMENSION_TO_TYPE[row["Dimension"]]
        windows = [item["L3_time"] for item in row["src_questions_info"]["Time"]]
        item = {
            "Question": row["Question"],
            "Options": row["Options"],
            "Answer": row["Answer"],
            "L3_time": windows,
            "Dimension": row["Dimension"],
            "src_questions_info": row["src_questions_info"],
        }
        # Content-derived IDs keep resume safe if rows are reordered. They
        # change when the question or its source windows change.
        identity = hashlib.sha256(
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if identity in output[qa_type]:
            raise ValueError(f"{source}[{index}]: duplicate question content")
        output[qa_type][identity] = item
    return output


def load_multi_point(path: str | Path, *, drama: str | None = None) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    """Load one extracted JSON file or one drama directly from DramaVQA.zip."""

    path = Path(path)
    if path.suffix.lower() == ".zip":
        if not drama or _DRAMA.fullmatch(drama) is None:
            raise ValueError("a safe --drama name is required when reading a ZIP")
        member = f"DramaVQA/Multi-Point/{drama}.json"
        with ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(member))
            except KeyError as exc:
                raise ValueError(f"ZIP has no {member}") from exc
        source = f"{path}!{member}"
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    summary = validate_multi_point(data, source=source)
    return normalize_multi_point(data, source=source), summary
