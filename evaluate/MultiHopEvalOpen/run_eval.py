"""Evaluate public multi-hop QA using its two source scenes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .qa_format import load_multi_hop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="one Multi-Hop JSON or DramaVQA.zip")
    parser.add_argument("--drama", help="drama name; required for ZIP input")
    parser.add_argument("--validate-only", action="store_true", help="check QA without images or a server")
    parser.add_argument("--image-root", type=Path, help="img_data root containing <drama>/<window>/render/")
    parser.add_argument("--output-dir", type=Path, help="directory for evaluation results")
    parser.add_argument("--model")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tpm-limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"))
    args = parser.parse_args(argv)
    drama = args.drama or (args.qa.stem if args.qa.suffix.lower() == ".json" else None)
    qa_data, summary = load_multi_hop(args.qa, drama=drama)
    print(json.dumps({"drama": drama, **summary}, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0
    if not args.image_root or not args.output_dir or not args.model:
        parser.error("--image-root, --output-dir and --model are required for evaluation")
    if args.limit < 0 or args.num_workers < 1:
        parser.error("--limit must be nonnegative and --num-workers must be positive")
    from MultiPointEvalOpen.legacy_evaluator import MultiPointEvaluator

    evaluator = MultiPointEvaluator(
        base_url=args.base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model=args.model,
        image_root=str(args.image_root),
        output_dir=str(args.output_dir),
        num_workers=args.num_workers,
        tpm_limit=args.tpm_limit,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        limit=args.limit,
        reasoning_effort=args.reasoning_effort,
    )
    evaluator.run(str(args.qa), drama, qa_data=qa_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
