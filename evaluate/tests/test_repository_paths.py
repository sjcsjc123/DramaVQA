"""Resolve tracked QA paths without depending on private workspace directories."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from dramavqa import cli
FIXTURE = Path(__file__).parent / 'fixtures/latest_qa.json'


def checkout(tmp_path, monkeypatch):
    root = tmp_path / 'repository'
    for group in ('Bench_Data', 'Bench_Data_mini_2000', 'Multi-Point', 'Multi-Hop', 'Training_Set'):
        (root / 'DramaVQA' / group).mkdir(parents=True)
    (root / 'evaluate/dramavqa').mkdir(parents=True)
    monkeypatch.setattr(cli, '__file__', str(root / 'evaluate/dramavqa/cli.py'))
    return root


@pytest.mark.parametrize('task,subset,group', [
    ('single-point', 'Bench_Data_mini_2000', 'Bench_Data_mini_2000'),
    ('single-point', 'Bench_Data', 'Bench_Data'),
    ('single-point', 'Training_Set', 'Training_Set'),
    ('multi-point', 'Bench_Data_mini_2000', 'Multi-Point'),
    ('multi-hop', 'Bench_Data_mini_2000', 'Multi-Hop'),
])
def test_explicit_root_selects_task_group(tmp_path, task, subset, group, monkeypatch):
    from unittest.mock import Mock
    expected = tmp_path / group / 'sample.json'
    expected.parent.mkdir()
    expected.write_text('{}')
    validator = Mock(return_value={'questions': 1})
    for name in ('validate_qa', 'validate_multi_point', 'validate_multi_hop'):
        monkeypatch.setattr(cli, name, validator)
    monkeypatch.setattr(cli, 'select_questions', lambda data, limit: {'selected': True})
    args = cli.build_parser().parse_args(['validate', '--task', task, '--subset', subset,
                '--drama', 'sample', '--data-root', str(tmp_path)])
    seen = []
    read_text = Path.read_text
    def read(path, *args, **kwargs):
        seen.append(path)
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    cli.load_input(args)
    assert seen == [expected]


@pytest.mark.parametrize('location', ['.', 'evaluate', 'evaluate/nested', 'outside'])
def test_checkout_discovery_and_source_fallback(tmp_path, monkeypatch, location, capsys):
    root = checkout(tmp_path, monkeypatch)
    (root / 'DramaVQA/Bench_Data_mini_2000/sample.json').write_bytes(FIXTURE.read_bytes())
    cwd = tmp_path / 'outside' if location == 'outside' else root / location
    cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(cwd)
    assert cli.main(['validate', '--task', 'single-point', '--drama', 'sample', '--limit', '1']) == 0
    assert json.loads(capsys.readouterr().out)['selected']['questions'] == 1


def test_explicit_qa_overrides_missing_data_root(tmp_path):
    assert cli.main(['validate', '--task', 'single-point', '--drama', 'sample',
                     '--qa', str(FIXTURE), '--data-root', str(tmp_path / 'missing')]) == 0


def test_missing_file_does_not_switch_subsets(tmp_path, monkeypatch, capsys):
    root = checkout(tmp_path, monkeypatch)
    (root / 'DramaVQA/Bench_Data_mini_2000/sample.json').write_bytes(FIXTURE.read_bytes())
    monkeypatch.chdir(root)
    with pytest.raises(SystemExit) as exc:
        cli.main(['validate', '--task', 'single-point', '--drama', 'sample', '--subset', 'Bench_Data'])
    assert exc.value.code == 2
    assert 'Bench_Data/sample.json' in capsys.readouterr().err


def test_installed_client_without_data_has_actionable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, '__file__', str(tmp_path / 'venv/lib/dramavqa/cli.py'))
    with pytest.raises(SystemExit) as exc:
        cli.main(['validate', '--task', 'single-point', '--drama', 'sample'])
    assert exc.value.code == 2
    assert '--data-root' in capsys.readouterr().err


def test_prepare_uses_selected_repository_questions(tmp_path, monkeypatch):
    root = checkout(tmp_path, monkeypatch)
    (root / 'DramaVQA/Bench_Data_mini_2000/sample.json').write_bytes(FIXTURE.read_bytes())
    monkeypatch.chdir(root)
    monkeypatch.setattr(cli.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0))
    output = root / 'runs/prepare'
    assert cli.main(['prepare', '--task', 'single-point', '--drama', 'sample', '--limit', '1',
                     '--video-root', str(root / 'media'), '--output-dir', str(output)]) == 1
    selected = json.loads((output / 'inputs/sample.json').read_text())
    assert sum(len(clip['Q&A']) for clip in selected.values()) == 1
    assert not json.loads((output / 'frames_summary.json').read_text())['complete']
