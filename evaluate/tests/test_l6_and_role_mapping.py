"""Regressions for full-drama CLI input and optional face-role maps."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from SinglePointEvalOpen.qa_format import load_qa, validate_qa
from dramavqa.cli import main


@pytest.mark.parametrize('module', ['SinglePointEvalOpen.legacy_compatible_runner',
                                    'AblationEvalOpen.ask_vllm'])
@pytest.mark.parametrize('mapping', [None, '{invalid', '[]', '{"Alice": "Alicia"}'])
def test_optional_role_map(module, mapping, tmp_path):
    cls = importlib.import_module(module).LLMInference
    engine = cls.__new__(cls)
    if mapping is not None:
        path = tmp_path / 'speaker_data/.workspace/new_roles_map.json'
        path.parent.mkdir(parents=True)
        path.write_text(mapping)
    assert engine._load_role_mapping(str(tmp_path)) == (
        {'Alice': 'Alicia'} if mapping == '{"Alice": "Alicia"}' else {})


def test_l6_is_preserved_and_checked_before_running(tmp_path, capsys):
    fixture = Path(__file__).parent / 'fixtures/latest_qa.json'
    data = json.loads(fixture.read_text())
    with pytest.raises(SystemExit):
        main(['validate', '--task', 'single-point', '--qa', str(fixture),
              '--drama', 'sample', '--level', 'L6'])
    assert 'L6_time/L6_frame' in capsys.readouterr().err
    for clip in data.values():
        clip.update(L6_time=clip['L5_time'], L6_frame=clip['L5_frame'])
    path = tmp_path / 'sample.json'
    path.write_text(json.dumps(data))
    records, summary = load_qa(path)
    assert 'L6' in summary['levels']
    assert records[0].time_for('L6') == data['42']['L6_time']
    assert main(['validate', '--task', 'single-point', '--qa', str(path),
                 '--drama', 'sample', '--level', 'L6']) == 0
    del data['42']['L6_frame']
    with pytest.raises(ValueError, match='L6_frame'):
        validate_qa(data)


@pytest.mark.parametrize('level', ['L3', 'L4', 'L5', 'L6'])
def test_public_video_to_ten_answers_without_role_map(tmp_path, level):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and ffprobe required')
    drama = 'breaking_bad'
    root = tmp_path / 'videos' / drama
    video = root / 'video_data'
    face = root / 'face_data'
    subtitle = root / 'subtitle_data'
    for p in (video, face, subtitle):
        p.mkdir(parents=True)
    for ep in (1, 2):
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', 'testsrc=size=64x64:rate=25:duration=2', '-c:v', 'ffv1',
                        str(video / f'{ep:04d}.mkv')], check=True)
        (face / f'{ep:04d}.json').write_text(json.dumps({'object_list': {
            str(i): {'start_frame': i, 'end_frame': i,
                     'id_list': {'0': {'name': 'Alice', 'score': 0.99}},
                     'geometry': {'active_frame_list': {'0': {'box': {
                         'left': 1, 'top': 20, 'right': 40, 'bottom': 50}}}}}
            for i in range(50)}}))
        (subtitle / f'{ep:04d}.json').write_text(json.dumps({'object_list': {'0': {
            'start_frame': 0, 'end_frame': 49, 'text': 'A test pattern'}}}))
    clip = {f'L{n}_time': '0001-00:00:00.000_0001-00:00:01.960' for n in range(2, 6)}
    clip.update({f'L{n}_frame': '0001-00000_0001-00049' for n in range(2, 6)})
    clip.update(L6_time='0001-00:00:00.000_0002-00:00:01.960',
                L6_frame='0001-00000_0002-00049')
    clip['Q&A'] = [dict(Question=f'What is visible {i}?', Options='A. Pattern\nB. Nothing',
                       Answer='A', Dimension='Object Presence') for i in range(12)]
    qa = tmp_path / 'questions.json'
    qa.write_text(json.dumps({'clip': clip}))
    output = tmp_path / 'out'
    common = ['--task', 'single-point', '--qa', str(qa), '--drama', drama, '--limit', '10',
              '--level', level, '--sample-frames', '4', '--target-width', '112',
              '--output-dir', str(output)]
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1]),
           'NO_PROXY': '127.0.0.1,localhost', 'OPENAI_API_KEY': 'test-only'}
    for name in ('API_MODEL_ID', 'VLLM_SERVED_MODEL_NAME'):
        env.pop(name, None)
    def command(stage, extra=()):
        result = subprocess.run([sys.executable, '-m', 'dramavqa', stage, *common, *extra],
                                env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    command('validate')
    command('prepare', ['--video-root', str(tmp_path / 'videos')])
    assert json.loads((output / 'frames_summary.json').read_text())['complete']
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = json.dumps({'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'message': {
                    'role': 'assistant', 'content': '<A>'}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command('run', ['--model', 'fixture', '--base-url',
                        f'http://127.0.0.1:{server.server_port}/v1'])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    report = json.loads((output / 'summary.json').read_text())
    assert report['complete'] and report['valid'] == report['recorded'] == 10
    assert report['guessed'] == report['truncated'] == 0
    assert len(seen) == 10
