"""Exercise the public command through a real local HTTP connection."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_public_multi_point_http_completion(tmp_path):
    pytest.importorskip("openai")
    from PIL import Image
    windows = ["0001-00:00:00.000_0001-00:00:01.000", "0002-00:00:00.000_0002-00:00:01.000"]
    qa = [{"Question": "Which scene is earlier?", "Options": "A. First\nB. Second", "Answer": "A",
        "Dimension": "Two-Event Temporal Comparison(TTC-2)",
        "src_questions_info": {"Question": ["First?", "Second?"], "Answers": ["one", "two"],
                               "Time": [{"L3_time": w} for w in windows]}}]
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(json.dumps(qa))
    output = tmp_path / "out"
    for window in windows:
        p = output / "images/L3/1/results.w112_nosub/img_data/sample" / window.replace(":", "-") / "render"
        p.mkdir(parents=True)
        Image.new("RGB", (112, 112), "red").save(p / "000000.jpg")
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"id": "test", "object": "chat.completion", "created": 1, "model": "fixture",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "<A>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])+os.pathsep+os.environ.get("PYTHONPATH", ""),
               "NO_PROXY": "127.0.0.1,localhost", "OPENAI_API_KEY": "test-only"}
        run = subprocess.run([sys.executable, "-m", "dramavqa", "run", "--task", "multi-point",
            "--qa", str(qa_path), "--drama", "sample", "--output-dir", str(output), "--model", "fixture",
            "--sample-frames", "1", "--target-width", "112", "--base-url", f"http://127.0.0.1:{server.server_port}/v1"],
            env=env, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
        report = json.loads((output / "summary.json").read_text())
        assert report["complete"] and report["recorded"] == 1
        assert len(seen) == 1
        assert sum(p["type"] == "image_url" for p in seen[0]["messages"][1]["content"]) == 2
        assert "test-only" not in (output / "run_manifest.json").read_text()
    finally:
        server.shutdown(); server.server_close(); thread.join()
