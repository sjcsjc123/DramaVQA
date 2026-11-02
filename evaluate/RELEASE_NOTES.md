# Standalone source release 0.2.4

This release exports the evaluation package into its own repository root.
It removes build artifacts, bytecode, deployment monitoring utilities that need
the historical workspace, and historical machine-specific validation reports.
The unified validate/prepare/run interface and original full prompt variants
remain available for single-point, multi-point, multi-hop, paired-source and
six ablation modes.

Standard installation now includes the OpenAI SDK, headless OpenCV, tqdm, NumPy
and Pillow. FFmpeg/ffprobe are system requirements. Model services, media,
annotations, fonts and the optional Conan backend are external inputs.
Gallery cache imports work through the installed package as well as the legacy
script entry point. Stale gallery error messages now point to the public CLI.
No prompt text, answer extraction rules or scoring rules are changed by this
standalone export.

## Validation

The source test suite passes 52 tests in a fresh Python 3.10 virtual environment
with system site packages disabled. Tests include generated video preparation,
a local HTTP fixture, QA adapters, result checks and the package gallery import
regression. Fixture-server tests do not validate a real model's answer quality.
The release is also checked by installing its wheel and running CLI commands
from outside the source tree without PYTHONPATH overrides. Real model trials
from the previous workspace have not been repeated for this export.

## Licensing and provenance

The destination DramaVQA repository has an existing MIT license. Its license
file is reproduced here so the evaluation directory can be distributed alone.
External media, annotations, fonts and model implementations are not bundled
and retain their own terms.
See COMPARISON.md for the retained historical evaluator origins and behavior.
