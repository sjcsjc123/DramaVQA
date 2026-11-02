"""Evaluation adapters for public DramaVQA multi-hop and paired source QA."""

from .qa_format import load_multi_hop, normalize_multi_hop, validate_multi_hop

__all__ = ["load_multi_hop", "normalize_multi_hop", "validate_multi_hop"]
