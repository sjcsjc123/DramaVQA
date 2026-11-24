"""A previous experiment's shell must not change the next evaluation."""
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image, features

from dramavqa import cli
from dramavqa.runtime import child_environment


STALE_SETTINGS = {
    'API_MODEL_ID': 'stale-model', 'VLLM_SERVED_MODEL_NAME': 'stale-served-model',
    'API_BASE_URL': 'http://unused.invalid/v1', 'OPENAI_BASE_URL': 'http://unused.invalid/v1',
    'API_KEY': 'obsolete-credential', 'SERVING_MODE': 'api',
    'ALLOW_MODEL_ID_MISMATCH': '1', 'ENABLE_THINKING': 'on', 'THINK_MODE': '1',
    'SYS_PROMPT_MODE': 'none', 'API_MAX_TOKENS': 'invalid', 'API_MAX_RETRIES': 'invalid',
    'API_TIMEOUT': 'invalid', 'API_TPM_LIMIT': 'invalid', 'LOCAL_API_TIMEOUT': 'invalid',
    'LOCAL_API_MAX_RETRIES': 'invalid', 'DECORD_NUM_THREADS': 'invalid',
    'GLM53_REASONING_EFFORT': 'invalid', 'EVAL_CLIENT_CONFIG': 'missing-policy.json',
    'EVAL_LOG_DIR': 'missing-log-dir', 'RUN_TAG': 'stale-run', 'SKIP_RETRY_GUESSED': '1',
    'RESTRICT_TO_QUESTION_SET': '0', 'ABLATE_BURN_SUBTITLE': '1', 'ABLATE_NO_SUBTITLE': '1',
    'OPENAI_ORG_ID': 'stale-org', 'OPENAI_PROJECT_ID': 'stale-project',
}


def test_subprocess_environment_keeps_connectivity_and_credentials_only():
    source = {**STALE_SETTINGS, 'OPENAI_API_KEY': 'test-only', 'PATH': 'runtime-path',
              'https_proxy': 'proxy', 'CUDA_VISIBLE_DEVICES': '2', 'LC_CTYPE': 'C.UTF-8'}
    assert child_environment(source) == {
        'OPENAI_API_KEY': 'test-only', 'PATH': 'runtime-path', 'https_proxy': 'proxy',
        'CUDA_VISIBLE_DEVICES': '2', 'LC_CTYPE': 'C.UTF-8'}
    assert source['API_MAX_TOKENS'] == 'invalid'


@pytest.mark.parametrize('task', cli.TASKS)
@pytest.mark.parametrize('entry', ['public', 'direct'])
@pytest.mark.parametrize('custom', [False, True])
def test_requests_ignore_stale_shell_settings(tmp_path, task, entry, custom):
    pytest.importorskip('openai')
    if custom and not features.check('webp'):
        pytest.skip('Pillow needs WebP codec support')
    windows = ['0001-00:00:00.000_0001-00:00:01.000',
               '0002-00:00:00.000_0002-00:00:01.000']
    choice = dict(Question='What is visible?', Options='A. Frame\nB. Nothing', Answer='A')
    if task == 'single-point':
        clip = {f'L{n}_time': windows[0] for n in range(2, 6)}
        clip.update({f'L{n}_frame': '0001-00000_0001-00001' for n in range(2, 6)})
        clip['Q&A'] = [{**choice, 'Dimension': 'Object Presence'}]
        data = {'clip': clip}
        windows = windows[:1]
    else:
        info = dict(Question=['First?', 'Second?'], Answers=['one', 'two'],
                    Time=[{'L3_time': window} for window in windows])
        dimension = 'Two-Event Temporal Comparison(TTC-2)'
        if task == 'multi-hop':
            info.update(Bridge_Character='Person 1', Options=[choice['Options']] * 2,
                        Answers=['A', 'A'], Dimension=['Object Presence'] * 2)
            dimension = 'Multi-Hop'
        data = [{**choice, 'Dimension': dimension, 'src_questions_info': info}]
    qa = tmp_path / 'sample.json'
    qa.write_text(json.dumps(data), encoding='utf-8')
    output = tmp_path / 'out'
    images = output / 'images'
    image_data = images / 'L3/1/results.w392_nosub/img_data'
    for window in windows:
        folder = image_data / 'sample' / window.replace(':', '-') / 'render'
        folder.mkdir(parents=True)
        Image.new('RGB', (392, 224), 'red').save(folder / '0001_0.jpg')
        (folder.parent / 'subtitles.json').write_text('{"0001_0.jpg":"Dialogue."}', encoding='utf-8')
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            seen.append((json.loads(self.rfile.read(int(self.headers['Content-Length']))), self.headers))
            body = json.dumps({'id': 'test', 'object': 'chat.completion', 'created': 1,
                'model': 'intended-model', 'choices': [{'index': 0, 'message': {
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
    base_url = f'http://127.0.0.1:{server.server_port}/v1'
    if entry == 'public':
        command = ['dramavqa', 'run', '--task', task, '--qa', str(qa), '--drama', 'sample',
            '--output-dir', str(output), '--model', 'intended-model', '--sample-frames', '1',
            '--target-width', '392', '--base-url', base_url]
    elif task == 'single-point':
        command = ['SinglePointEvalOpen.legacy_compatible_runner', '--question_path', str(qa),
            '--drama', 'sample', '--image_path', str(images), '--result_path', str(output / 'results'),
            '--model_name', 'intended-model', '--sample_frames', '1', '--target_width', '392',
            '--use_render_info', '--base-url', base_url]
    else:
        module = 'MultiPointEvalOpen.run_eval' if task == 'multi-point' else 'MultiHopEvalOpen.run_eval'
        command = [module, '--qa', str(qa), '--drama', 'sample', '--image-root', str(image_data),
            '--output-dir', str(output / 'results'), '--model', 'intended-model', '--base-url', base_url]
    if custom:
        command += ['--max-tokens', '1234', '--temperature', '0.3', '--top-p', '0.2',
                    '--max-retries', '1', '--api-timeout', '17', '--tpm-limit', '99999',
                    '--reasoning-effort', 'low', '--run-tag', 'explicit-run',
                    '--evaluation-profile', 'local-glm', '--image-encoding', 'webp-lossless',
                    '--thinking-transport', 'enable_thinking', '--decode-threads', '3',
                    '--enable_thinking' if entry == 'direct' and task == 'single-point'
                    else '--model-thinking', 'on']
        if task == 'single-point':
            command += ['--generation-policy', 'strict', '--thinking-token-budget', '512',
                        '--guess-after-failed-retries', '2', '--no-retry-guessed']
    env = {**os.environ, **STALE_SETTINGS, 'OPENAI_API_KEY': 'test-only',
           'PYTHONPATH': str(Path(__file__).parents[1]), 'PYTHONUTF8': '1',
           'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost'}
    try:
        run = subprocess.run([sys.executable, '-m', *command], env=env,
                             capture_output=True, text=True, encoding='utf-8', timeout=45)
        assert run.returncode == 0, run.stdout + run.stderr
        assert len(seen) == 1
        body, headers = seen[0]
        assert body['model'] == 'intended-model'
        assert headers['Authorization'] == 'Bearer test-only'
        assert not headers.get('OpenAI-Organization') and not headers.get('OpenAI-Project')
        assert body['max_tokens'] == (1234 if custom else 1000 if task == 'single-point' else 4000)
        assert body['temperature'] == (0.3 if custom else 0.1)
        assert body['top_p'] == (0.2 if custom else 0.01)
        if custom:
            assert body['enable_thinking'] is True and body['reasoning_effort'] == 'low'
            if task == 'single-point':
                assert body['thinking_token_budget'] == 512
        else:
            assert body['chat_template_kwargs'] == {'enable_thinking': False}
            assert 'reasoning_effort' not in body
        parts = body['messages'][1]['content']
        url = next(p['image_url']['url'] for p in parts if p['type'] == 'image_url')
        assert url.startswith('data:image/webp;' if custom else 'data:image/png;')
        with Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))) as image:
            assert image.size == ((392, 224) if custom else (384, 224))
        assert any('Dialogue.' in p.get('text', '') for p in parts)
        assert 'subtitles rendered onto the images' not in body['messages'][0]['content']
        meta_file = (next((output / 'results').rglob('run_meta.json')) if task == 'single-point'
                     else output / 'results/evaluation_settings.json')
        meta = json.loads(meta_file.read_text(encoding='utf-8'))
        assert meta['run_tag'] == ('explicit-run' if custom else '')
        assert meta['max_retries'] == (1 if custom else 6)
        assert meta['api_timeout'] == (17 if custom else 120)
        assert meta['base_url'] == base_url
        assert 'test-only' not in meta_file.read_text(encoding='utf-8')
        if task == 'single-point':
            assert meta['think_mode'] is False
            assert meta['enable_thinking_sent'] is custom
            assert meta['retry_guessed'] is not custom
            assert meta['generation_policy'] == ('strict' if custom else 'baseline')
        if custom and entry == 'direct':
            changed = subprocess.run([sys.executable, '-m', *command, '--temperature', '0.4'],
                env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
            assert changed.returncode != 0 and len(seen) == 1
            assert 'settings changed' in changed.stdout + changed.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize('flag,value', [('--max-retries', '2'), ('--num-workers', '2'),
    ('--api-timeout', '18'), ('--temperature', '0.2'), ('--top-p', '0.5'),
    ('--model-thinking', 'on'), ('--reasoning-effort', 'low'), ('--run-tag', 'another')])
def test_explicit_changes_cannot_mix_results(tmp_path, monkeypatch, flag, value):
    calls = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda *a, **k:
                        calls.append(a) or SimpleNamespace(returncode=0))
    args = ['run', '--task', 'single-point', '--drama', 'sample', '--model', 'fixture',
            '--qa', str(Path(__file__).parent / 'fixtures/latest_qa.json'), '--output-dir', str(tmp_path)]
    cli.main(args)
    with pytest.raises(SystemExit) as exc:
        cli.main(args + [flag, value])
    assert exc.value.code == 2 and len(calls) == 1


@pytest.mark.parametrize('task', ['multi-point', 'multi-hop'])
def test_unsupported_policies_fail_before_launch(task):
    with pytest.raises(SystemExit) as exc:
        cli.main(['run', '--task', task, '--drama', 'sample', '--generation-policy', 'strict'])
    assert exc.value.code == 2
