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

## Updated question data and merged media directories (unreleased)

Replace all 51 question files with the supplied updated annotations and English
drama identifiers. All media folders now match QA filename stems. English-language
series use one directory per drama and continuous episode numbers across seasons;
no season offset is inferred. Update single-point, ablation, offline preparation,
speaker metadata lookup, Conan defaults, and documentation accordingly.

Validation: 72 automated tests, including synthetic extraction from episode 9 and
scene ordering across episodes 8 and 9; all 51 QA files (113,136 questions) validate.
Real media merging and model inference are outside this change.

## Repository question paths

Question annotations live in the GitHub checkout's `DramaVQA/` directory.
The unified CLI can discover that directory or accept `--data-root`, then select
the task's group and exact drama filename. Single-point and ablation default to
Mini; full and training subsets remain explicit. `--qa` still overrides lookup
and accepts JSON or ZIP. Media paths remain separate and explicitly configured.
Documentation examples now run from the repository root and reference tracked
question JSON files and the documented Hugging Face media download directory.

The updated suite passes 68 tests. All 51 tracked question files (113,136
questions) pass public CLI validation. An installed wheel resolves data from the
repository root and `evaluate/`, and accepts explicit data roots outside the
checkout. These checks do not load models or download media.

## Standalone 0.2.4 validation

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
