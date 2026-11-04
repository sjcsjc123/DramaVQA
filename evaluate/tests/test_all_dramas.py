"""Batch discovery, failure accounting, and per-drama video-to-HTTP evaluation."""
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zipfile import ZipFile

import pytest

from dramavqa.cli import main

FIXTURE = Path(__file__).parent / 'fixtures/latest_qa.json'


@pytest.mark.parametrize('source', ['root', 'directory', 'zip'])
def test_all_selects_only_requested_collection_and_limits_per_drama(tmp_path, source):
    root = tmp_path / 'questions'
    mini = root / 'Bench_Data_mini_2000'
    mini.mkdir(parents=True)
    (root / 'Training_Set').mkdir()
    for drama in ('z_show', 'a_show'):
        (mini / f'{drama}.json').write_bytes(FIXTURE.read_bytes())
    (root / 'Training_Set/train.json').write_text('{}')
    video_root = tmp_path / 'videos'
    for name in ('a_show', 'z_show'):
        (video_root / name / 'video_data').mkdir(parents=True)
    (mini / 'not_downloaded.json').write_bytes(FIXTURE.read_bytes())
    args = ['--data-root', str(root)]
    if source == 'directory':
        args = ['--qa', str(mini)]
    elif source == 'zip':
        path = tmp_path / 'questions.zip'
        with ZipFile(path, 'w') as archive:
            for p in root.rglob('*.json'):
                archive.write(p, 'DramaVQA/' + str(p.relative_to(root)))
        args = ['--qa', str(path)]
    output = tmp_path / 'output'
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert main(['validate', '--task', 'single-point', '--drama', 'all',
                     '--limit', '2', '--video-root', str(video_root), '--output-dir', str(output), *args]) == 0
    report = json.loads((output / 'batch_validate_summary.json').read_text())
    assert report['dramas'] == report['successful'] == 2
    assert [r['drama'] for r in report['results']] == ['a_show', 'z_show']
    selections = [json.loads(line)['selected']['questions'] for line in stream.getvalue().splitlines()
                  if line.startswith('{"source":')]
    assert selections == [2, 2]


def test_batch_validation_continues_and_reports_failure(tmp_path):
    folder = tmp_path / 'qa'
    folder.mkdir()
    (folder / 'a_broken.json').write_text('{}')
    (folder / 'z_good.json').write_bytes(FIXTURE.read_bytes())
    output = tmp_path / 'out'
    video_root = tmp_path / 'videos'
    for name in ('a_broken', 'z_good'):
        (video_root / name / 'video_data').mkdir(parents=True)
    assert main(['validate', '--task', 'single-point', '--qa', str(folder),
                 '--drama', 'all', '--video-root', str(video_root), '--output-dir', str(output)]) == 1
    report = json.loads((output / 'batch_validate_summary.json').read_text())
    assert report['failed'] == report['successful'] == 1
    assert not report['complete']
    assert report['results'][1]['complete']


def test_all_rejects_single_json_and_empty_collection(tmp_path):
    for source in (FIXTURE, tmp_path):
        with pytest.raises(SystemExit) as exc:
            main(['validate', '--task', 'single-point', '--qa', str(source), '--drama', 'all'])
        assert exc.value.code == 2


@pytest.mark.parametrize('task', ['single-point', 'multi-point', 'multi-hop'])
def test_all_dramas_isolated_video_to_http_pipeline(tmp_path, task):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and ffprobe required')
    group = {'single-point': 'Bench_Data_mini_2000', 'multi-point': 'Multi-Point',
             'multi-hop': 'Multi-Hop'}[task]
    qa_dir = tmp_path / 'questions' / group
    qa_dir.mkdir(parents=True)
    video_root = tmp_path / 'videos'
    window = '0001-00:00:00.000_0001-00:00:01.960'
    question = dict(Question='What is visible?', Options='A. Pattern\nB. Nothing',
                    Answer='A', Dimension='Object Presence')
    for drama in ('a_show', 'z_show'):
        video = video_root / drama / 'video_data' / '0001.mkv'
        video.parent.mkdir(parents=True)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
                        'testsrc=size=64x64:rate=25:duration=2', '-c:v', 'ffv1', str(video)], check=True)
        if task == 'single-point':
            clip = {f'L{n}_time': window for n in range(2, 7)}
            clip.update({f'L{n}_frame': '0001-00000_0001-00049' for n in range(2, 7)})
            clip['Q&A'] = [question.copy() for _ in range(3)]
            data = {'clip': clip}
        else:
            row = dict(question)
            if task == 'multi-point':
                row['Dimension'] = 'Two-Event Temporal Comparison(TTC-2)'
            row['src_questions_info'] = dict(Bridge_Character='Alice',
                Question=['Scene one?', 'Scene two?'], Options=[question['Options']] * 2,
                Answers=['A', 'B'], Dimension=['Object Presence'] * 2,
                Time=[{'L3_time': window}, {'L3_time': window}])
            data = [dict(row, Question=f'What is visible {i}?') for i in range(3)]
        (qa_dir / f'{drama}.json').write_text(json.dumps(data))
    (qa_dir / 'not_downloaded.json').write_text(json.dumps(data))
    output = tmp_path / 'out'
    common = ['--task', task, '--drama', 'all', '--data-root', str(tmp_path / 'questions'),
              '--limit', '2', '--sample-frames', '4', '--target-width', '112',
              '--output-dir', str(output)]
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).parents[1]),
           'NO_PROXY': '127.0.0.1,localhost', 'OPENAI_API_KEY': 'test-only'}
    for key in ('API_MODEL_ID', 'VLLM_SERVED_MODEL_NAME'):
        env.pop(key, None)
    def command(stage, extra=()):
        result = subprocess.run([sys.executable, '-m', 'dramavqa', stage, *common, *extra],
                                env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    command('validate', ['--video-root', str(video_root)])
    command('prepare', ['--video-root', str(video_root)])
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
    report = json.loads((output / 'batch_run_summary.json').read_text())
    assert report['complete'] and report['successful'] == 2
    assert len(seen) == 4
    for row in report['results']:
        assert row['summary']['valid'] == row['summary']['recorded'] == 2
        path = output / row['drama']
        assert (path / 'input_manifest.json').is_file()
        assert (path / 'summary.json').is_file()
        assert list((path / 'images').rglob('*.jpg'))


def test_all_requires_local_media_or_prepared_list(tmp_path):
    directory = tmp_path / 'qa'
    directory.mkdir()
    (directory / 'a_show.json').write_bytes(FIXTURE.read_bytes())
    with pytest.raises(SystemExit) as exc:
        main(['validate', '--task', 'single-point', '--qa', str(directory), '--drama', 'all'])
    assert exc.value.code == 2
    empty_media = tmp_path / 'videos'
    empty_media.mkdir()
    with pytest.raises(SystemExit):
        main(['validate', '--task', 'single-point', '--qa', str(directory),
              '--drama', 'all', '--video-root', str(empty_media)])
