"""Evaluate public DramaVQA Multi-Point QA with the historical scoring flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .qa_format import load_multi_point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True, help="one Multi-Point JSON or DramaVQA.zip")
    parser.add_argument("--drama", help="drama name; required for ZIP input")
    parser.add_argument("--validate-only", action="store_true", help="check QA without images or a server")
    parser.add_argument("--image-root", type=Path, help="img_data root containing <drama>/<window>/render/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument('--evaluation-profile', choices=('local-qwen', 'local-glm'), default='local-qwen')
    parser.add_argument('--model-thinking', choices=('on', 'off', 'omit'), default='off')
    parser.add_argument('--thinking-transport', choices=('chat-template', 'enable_thinking', 'thinking'), default='chat-template')
    parser.add_argument('--image-encoding', choices=('png', 'webp-lossless'), default='png')
    from dramavqa.runtime import (add_execution_arguments, execution_settings,
                                  validate_execution_settings, validate_task_settings)
    add_execution_arguments(parser)
    args = parser.parse_args(argv)
    if args.max_tokens is None:
        args.max_tokens = 4000
    try:
        validate_execution_settings(args)
        validate_task_settings(args, multi=True)
    except ValueError as exc:
        parser.error(str(exc))
    drama = args.drama or (args.qa.stem if args.qa.suffix.lower() == ".json" else None)
    qa_data, summary = load_multi_point(args.qa, drama=drama)
    print(json.dumps({"drama": drama, **summary}, ensure_ascii=False), flush=True)
    if args.validate_only:
        return 0
    if not args.image_root or not args.output_dir or not args.model:
        parser.error("--image-root, --output-dir and --model are required for evaluation")
    if args.limit < 0 or args.num_workers < 1:
        parser.error("--limit must be nonnegative and --num-workers must be positive")
    settings = dict(evaluation_profile=args.evaluation_profile, model_thinking=args.model_thinking,
                    image_encoding=args.image_encoding, thinking_transport=args.thinking_transport, max_tokens=args.max_tokens,
                    model=args.model, base_url=args.base_url, api_timeout=args.api_timeout,
                    num_workers=args.num_workers, **execution_settings(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'evaluation_settings.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != settings:
        parser.error('Evaluation settings changed; use a new output directory')
    manifest.write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    from .legacy_evaluator import MultiPointEvaluator

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
        evaluation_profile=args.evaluation_profile, model_thinking=args.model_thinking,
        image_encoding=args.image_encoding, thinking_transport=args.thinking_transport,
        temperature=args.temperature, top_p=args.top_p,
    )
    evaluator.run(str(args.qa), drama, qa_data=qa_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
