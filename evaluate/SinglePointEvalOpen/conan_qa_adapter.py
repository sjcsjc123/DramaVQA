"""Convert current flat single-point QA to Conan's legacy nested QA input.

The Conan evaluator expects ``category -> clip_id -> Q&A dict``. This adapter
changes only that input shape; it does not alter questions, options, answers,
time windows, images, or Conan's inference code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .qa_format import validate_qa
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from SinglePointEvalOpen.qa_format import validate_qa


def convert(data: dict, *, category: str = "single-point") -> dict:
    validate_qa(data)
    nested = {}
    for clip_id, clip in data.items():
        if not clip["Q&A"]:
            continue
        nested[clip_id] = {**clip, "Q&A": {str(i): question for i, question in enumerate(clip["Q&A"])}}
    return {category: nested}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="current flat QA JSON")
    parser.add_argument("output", type=Path, help="Conan-compatible QA JSON")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    converted = convert(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"converted {sum(len(clip['Q&A']) for clip in converted['single-point'].values())} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
