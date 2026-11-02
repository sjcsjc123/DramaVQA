"""Adapters for the six historical single-point ablation modes."""

from .qa_adapter import convert_ablation_qa, load_ablation_qa

__all__ = ["convert_ablation_qa", "load_ablation_qa"]
