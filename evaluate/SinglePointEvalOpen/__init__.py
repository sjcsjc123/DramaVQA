"""Single-point QA validation and the historical-compatible evaluator.

The package accepts the flat QA files distributed with the current benchmark.
Multi-hop data is intentionally outside this package.
"""

from .qa_format import QARecord, load_qa, validate_qa
__all__ = ["QARecord", "load_qa", "validate_qa"]
