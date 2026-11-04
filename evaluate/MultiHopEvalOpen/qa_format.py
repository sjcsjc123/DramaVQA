"""Adapt public Multi-Hop rows to the multi-scene evaluator input."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

from MultiPointEvalOpen.qa_format import _OPTION, _text, _window


_DRAMA = re.compile(r"[A-Za-z0-9_.-]+")


def _choice(question: Any, options: Any, answer: Any, where: str) -> None:
    _text(question, "Question", where)
    options = _text(options, "Options", where)
    answer = _text(answer, "Answer", where)
    letters = {match.group(1) for match in _OPTION.finditer(options)}
    if len(answer) != 1 or answer not in letters:
        raise ValueError(f"{where}: Answer {answer!r} is absent from Options")


def validate_multi_hop(data: Any, *, source: str = "Multi-Hop") -> dict[str, Any]:
    if not isinstance(data, list):
        raise ValueError(f"{source}: expected a list of questions")
    dimensions: Counter[str] = Counter()
    for index, row in enumerate(data):
        where = f"{source}[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where}: expected an object")
        _choice(row.get("Question"), row.get("Options"), row.get("Answer"), where)
        dimensions[_text(row.get("Dimension"), "Dimension", where)] += 1
        info = row.get("src_questions_info")
        if not isinstance(info, dict):
            raise ValueError(f"{where}: src_questions_info must be an object")
        _text(info.get("Bridge_Character"), "Bridge_Character", where)
        for field in ("Question", "Options", "Answers", "Dimension", "Time"):
            values = info.get(field)
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f"{where}: src_questions_info.{field} must contain two items")
        for hop in (0, 1):
            source_where = f"{where} source[{hop}]"
            _choice(info["Question"][hop], info["Options"][hop], info["Answers"][hop], source_where)
            _text(info["Dimension"][hop], "Dimension", source_where)
            time = info["Time"][hop]
            if not isinstance(time, dict):
                raise ValueError(f"{source_where}: Time must be an object")
            _window(time.get("L3_time"), source_where)
    return {"questions": len(data), "dimensions": dict(dimensions)}


def normalize_multi_hop(data: Any, *, source: str = "Multi-Hop") -> dict[str, dict[str, dict[str, Any]]]:
    """Produce the historical ``multi_hop -> id -> QA`` container."""

    validate_multi_hop(data, source=source)
    output: dict[str, dict[str, dict[str, Any]]] = {"multi_hop": {}}
    for index, row in enumerate(data):
        info = row["src_questions_info"]
        question, options, answer, dimension = (
            row["Question"], row["Options"], row["Answer"], row["Dimension"]
        )
        item = {
            "Question": question,
            "Options": options,
            "Answer": answer,
            "L3_time": [time["L3_time"] for time in info["Time"]],
            "Dimension": dimension,
            "src_questions_info": info,
        }
        # Include the entire paired row so repeated source questions with
        # different partner scenes remain separate evaluation inputs.
        identity = hashlib.sha256(
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if identity in output["multi_hop"]:
            raise ValueError(f"{source}[{index}]: duplicate paired row")
        output["multi_hop"][identity] = item
    return output


def load_multi_hop(path: str | Path, *, drama: str | None = None) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        if not drama or _DRAMA.fullmatch(drama) is None:
            raise ValueError("a safe --drama name is required when reading a ZIP")
        member = f"DramaVQA/Multi-Hop/{drama}.json"
        with ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(member))
            except KeyError as exc:
                raise ValueError(f"ZIP has no {member}") from exc
        source = f"{path}!{member}"
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    summary = validate_multi_hop(data, source=source)
    return normalize_multi_hop(data, source=source), summary
