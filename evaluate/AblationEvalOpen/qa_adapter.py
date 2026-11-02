"""Adapt current flat single-point QA to the historical ablation container."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

from SinglePointEvalOpen.qa_format import validate_qa


_DRAMA = re.compile(r"[A-Za-z0-9_.-]+")


def convert_ablation_qa(data: Any) -> dict[str, dict[str, Any]]:
    """Keep all questions and windows, changing only the outer QA shape."""

    validate_qa(data)
    clips = {}
    for clip_id, clip in data.items():
        if clip["Q&A"]:
            clips[clip_id] = {
                **clip,
                "Q&A": {str(index): question for index, question in enumerate(clip["Q&A"])},
            }
    return {"low-level": clips}


def load_ablation_qa(path: str | Path, *, drama: str | None = None,
                     subset: str = "Bench_Data_mini_2000") -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        if subset not in ("Bench_Data", "Bench_Data_mini_2000"):
            raise ValueError(f"unsupported ZIP subset {subset!r}")
        if not drama or _DRAMA.fullmatch(drama) is None:
            raise ValueError("a safe --drama name is required when reading a ZIP")
        member = f"DramaVQA/{subset}/{drama}.json"
        with ZipFile(path) as archive:
            try:
                data = json.loads(archive.read(member))
            except KeyError as exc:
                raise ValueError(f"ZIP has no {member}") from exc
        source = f"{path}!{member}"
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(path)
    summary = validate_qa(data, source=source)
    return convert_ablation_qa(data), summary
