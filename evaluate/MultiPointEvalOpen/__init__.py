"""Evaluation adapter for the public DramaVQA multi-point QA format."""

from .qa_format import load_multi_point, normalize_multi_point, validate_multi_point

__all__ = ["load_multi_point", "normalize_multi_point", "validate_multi_point"]
