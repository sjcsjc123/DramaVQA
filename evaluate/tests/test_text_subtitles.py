"""Subtitle instructions must match image-then-text messages on every public path."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from dramavqa import cli
from SinglePointEvalOpen import legacy_compatible_runner as sampler


@pytest.mark.parametrize('task', ['single-point', 'multi-point', 'multi-hop'])
@pytest.mark.parametrize('mode', ['text', 'burn', 'none'])
def test_removed_public_subtitle_switch_is_rejected(task, mode, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(['validate', '--task', task, '--drama', 'sample',
                                      '--subtitles', mode])
    assert exc.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('flag', ['--subtitle_text', '--render_subtitle', '--no_render_subtitle'])
def test_removed_internal_subtitle_switches_are_rejected(flag, capsys):
    with pytest.raises(SystemExit) as exc:
        sampler.main(['--drama', 'sample', '--model_name', 'fixture',
                      '--question_path', 'qa.json', flag])
    assert exc.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
def test_single_point_request_and_metadata_use_text_subtitles(tmp_path, monkeypatch, parallel, thinking):
    window = '0001-00:00:00.000_0001-00:00:01.000'
    clip = {f'L{n}_time': window for n in range(2, 6)}
    clip.update({f'L{n}_frame': '0001-00000_0001-00001' for n in range(2, 6)})
    clip['Q&A'] = [dict(Question='What is visible?', Options='A. Frame\nB. Nothing',
                       Answer='A', Dimension='Object Presence')]
    qa = tmp_path / 'sample.json'
    qa.write_text(json.dumps({'clip': clip}), encoding='utf-8')
    image_root = tmp_path / 'images'
    folder = (image_root / 'L3/2/results.w112_nosub/img_data/sample'
              / window.replace(':', '-') / 'render')
    folder.mkdir(parents=True)
    subtitles = {'0001_0.jpg': 'First dialogue.', '0001_1.jpg': 'Second dialogue.'}
    for name in subtitles:
        Image.new('RGB', (112, 112), 'red').save(folder / name)
    (folder.parent / 'subtitles.json').write_text(json.dumps(subtitles), encoding='utf-8')
    requests = []

    class RecordingEvaluator(sampler.LLMInference):
        def get_llm_response(self, messages, key, **kwargs):
            requests.append(messages)
            return 'A', 0, '<A>', None

    monkeypatch.setattr(sampler, 'LLMInference', RecordingEvaluator)
    monkeypatch.setattr(sampler, 'OpenAI', lambda **kwargs: SimpleNamespace())
    # Old experiment environment switches cannot alter the fixed text input policy.
    monkeypatch.setenv('ABLATE_BURN_SUBTITLE', '1')
    monkeypatch.setenv('ABLATE_NO_SUBTITLE', '1')
    args = ['--drama', 'sample', '--model_name', 'fixture',
            '--question_path', str(qa), '--image_path', str(image_root),
            '--result_path', str(tmp_path / 'results'), '--use_render_info',
            '--sample_frames', '2', '--target_width', '112', '--level', 'L3', '--num_workers', '1',
            '--thinking', 'on' if thinking else 'off']
    sampler.main(args + (['--parallel'] if parallel else []))
    assert len(requests) == 1
    system = requests[0][0]['content']
    assert 'provided as text right after that frame' in system
    assert 'subtitles rendered onto the images' not in system
    content = requests[0][1]['content']
    assert [p['type'] for p in content] == ['text', 'image_url', 'text', 'image_url', 'text']
    assert content[2]['text'] == '字幕: First dialogue.'
    assert content[4]['text'] == '字幕: Second dialogue.'
    metadata_paths = list((tmp_path / 'results').rglob('run_meta.json'))
    assert len(metadata_paths) == 1
    metadata = json.loads(metadata_paths[0].read_text(encoding='utf-8'))
    assert metadata['system_prompt'] == system
    assert metadata['subtitle_format'] == 'text'


def test_old_prompt_run_cannot_be_silently_resumed(tmp_path, monkeypatch):
    fixture = Path(__file__).parent / 'fixtures/latest_qa.json'
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, 'run', run)
    args = ['run', '--task', 'single-point', '--qa', str(fixture), '--drama', 'sample',
            '--output-dir', str(tmp_path), '--model', 'fixture']
    cli.main(args)
    manifest = tmp_path / 'run_manifest.json'
    previous = json.loads(manifest.read_text(encoding='utf-8'))
    previous.pop('prompt_revision')
    manifest.write_text(json.dumps(previous), encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    assert len(calls) == 1
