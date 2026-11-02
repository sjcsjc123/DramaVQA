"""Reader and validator for the current flat single-point QA format.

Current files have this shape::

    {
      "clip_id": {
        "L2_time": "S1#0001-..._S1#0001-...",
        "L3_time": "...",
        "L4_time": "...",
        "L5_time": "...",
        "L2_frame": "...",
        "L3_frame": "...",
        "L4_frame": "...",
        "L5_frame": "...",
        "Q&A": [{"Question": "...", "Options": "...", "Answer": "A", "Dimension": "..."}]
      }
    }

One clip may contain several independent questions.  The evaluator therefore
emits one :class:`QARecord` per item in ``Q&A`` and keeps the clip window shared.
No category guessing or multi-hop conversion is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


LEVELS = ("L2", "L3", "L4", "L5")
_TIME_PART = re.compile(
    r"^(?:S(?P<season>\d+)#)?(?P<episode>\d+)-(?P<clock>\d{2}:\d{2}:\d{2}\.\d{3})$"
)


@dataclass(frozen=True)
class QARecord:
    """A normalized single-point question."""

    clip_id: str
    question_id: str
    question: str
    options: str
    answer: str
    dimension: str
    times: Mapping[str, str]
    frames: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def time_for(self, level: str) -> str:
        try:
            return self.times[level]
        except KeyError as exc:
            raise ValueError(f"QA record {self.question_id} has no {level}_time") from exc


def _require_text(value: Any, name: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {name} must be a non-empty string")
    return value.strip()


def _validate_time(value: str, where: str) -> None:
    parts = value.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"{where}: time range must contain one '_' separator")
    parsed = []
    for part in parts:
        match = _TIME_PART.fullmatch(part)
        if not match:
            raise ValueError(f"{where}: invalid time endpoint {part!r}")
        parsed.append(match)
    if parsed[0].group("season") and parsed[1].group("season"):
        if parsed[0].group("season") != parsed[1].group("season"):
            # Cross-season windows are valid for L5 in the supplied data.
            pass


def _option_letters(options: str) -> set[str]:
    return {
        match.group(1).upper()
        for match in re.finditer(r"^\s*\**\s*([A-Z])\s*[.、,，):：]", options, re.MULTILINE)
    }


def validate_qa(data: Any, *, source: str = "QA") -> dict[str, Any]:
    """Validate a current-format QA object and return summary counts.

    The function is deliberately strict about the fields used by evaluation.
    Extra fields are allowed so future metadata does not break old runners.
    """

    if not isinstance(data, dict) or not data:
        raise ValueError(f"{source}: expected a non-empty object keyed by clip id")
    clips = questions = 0
    for clip_id, clip in data.items():
        where = f"{source} clip {clip_id!r}"
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"{where}: clip id must be a non-empty string")
        if not isinstance(clip, dict):
            raise ValueError(f"{where}: expected an object")
        for level in LEVELS:
            time_key = f"{level}_time"
            frame_key = f"{level}_frame"
            _validate_time(_require_text(clip.get(time_key), time_key, where), f"{where} {time_key}")
            _require_text(clip.get(frame_key), frame_key, where)
        qa = clip.get("Q&A")
        if not isinstance(qa, list):
            raise ValueError(f"{where}: Q&A must be a list")
        for q_index, item in enumerate(qa):
            qwhere = f"{where} Q&A[{q_index}]"
            if not isinstance(item, dict):
                raise ValueError(f"{qwhere}: expected an object")
            question = _require_text(item.get("Question"), "Question", qwhere)
            options = _require_text(item.get("Options"), "Options", qwhere)
            answer = _require_text(item.get("Answer"), "Answer", qwhere).upper()
            letters = _option_letters(options)
            if letters and answer.split()[0][:1] not in letters:
                raise ValueError(f"{qwhere}: answer {answer!r} is absent from options {sorted(letters)}")
            _require_text(item.get("Dimension"), "Dimension", qwhere)
            questions += 1
        clips += 1
    return {"clips": clips, "questions": questions, "levels": list(LEVELS)}


def load_qa(path: str | Path) -> tuple[list[QARecord], dict[str, Any]]:
    """Load and normalize a current-format JSON file."""

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_qa(data, source=str(path))
    records: list[QARecord] = []
    for clip_id, clip in data.items():
        times = {level: clip[f"{level}_time"] for level in LEVELS}
        frames = {level: clip[f"{level}_frame"] for level in LEVELS}
        for q_index, item in enumerate(clip["Q&A"]):
            records.append(
                QARecord(
                    clip_id=str(clip_id),
                    question_id=f"{clip_id}:{q_index}",
                    question=item["Question"].strip(),
                    options=item["Options"].strip(),
                    answer=item["Answer"].strip().split()[0][:1].upper(),
                    dimension=item["Dimension"].strip(),
                    times=times,
                    frames=frames,
                )
            )
    return records, summary


def iter_records(paths: Iterable[str | Path]) -> Iterable[QARecord]:
    for path in paths:
        records, _ = load_qa(path)
        yield from records
