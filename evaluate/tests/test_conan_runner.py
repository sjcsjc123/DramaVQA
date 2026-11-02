"""Exercise the Conan handoff without loading its GPU model."""

import json
from pathlib import Path
import sys

try:
    from SinglePointEvalOpen.conan_runner import main
except ModuleNotFoundError:
    from conan_runner import main


FIXTURE = Path(__file__).parent / "fixtures/latest_qa.json"


def test_conan_launcher_converts_qa_and_calls_backend(tmp_path):
    backend = tmp_path / "Conan-Eval"
    backend.mkdir()
    (backend / "singlehop.py").write_text("# backend marker\n", encoding="utf-8")
    (backend / "main_singlehop.py").write_text(
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "assert args[args.index('--benchmark') + 1] == 'singlehop'\n"
        "assert args[args.index('--mode') + 1] == 'step'\n"
        "assert args[args.index('--max_steps') + 1] == '3'\n"
        "assert os.environ['CONAN_USE_DRAMA_QA'] == '1'\n"
        "assert os.environ['CONAN_QA_LEVEL'] == 'L3'\n"
        "qa = pathlib.Path(os.environ['CONAN_QUESTIONS_DIR']) / 'sample.json'\n"
        "data = json.loads(qa.read_text())\n"
        "assert sum(len(c['Q&A']) for c in data['single-point'].values()) == 3\n"
        "pathlib.Path(os.environ['CONAN_OUTPUT_ROOT'], 'backend_ok').write_text('ok')\n",
        encoding="utf-8",
    )
    images = tmp_path / "images"
    model = tmp_path / "model"
    images.mkdir()
    model.mkdir()
    output = tmp_path / "output"
    args = ["--qa", str(FIXTURE), "--image-root", str(images),
            "--output-root", str(output), "--conan-eval", str(backend),
            "--model-path", str(model), "--python", sys.executable,
            "--drama", "sample", "--level", "L3", "--sample-frames", "8"]
    assert main(args + ["--dry-run"]) == 0
    assert not output.exists()
    assert main(args) == 0
    assert (output / "backend_ok").read_text() == "ok"
    converted = json.loads((output / "inputs/conan_qa/sample.json").read_text())
    assert list(converted) == ["single-point"]
