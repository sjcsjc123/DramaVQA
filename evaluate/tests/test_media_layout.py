"""Exercise flat metadata reads and legacy media layout compatibility."""
import json
import threading

import pytest

from SinglePointEvalOpen.media_layout import resolve_data_dir
from AblationEvalOpen.ask_vllm import LLMInference


@pytest.mark.parametrize('layout', ['flat', 'manifest', 'subdirectory'])
def test_gallery_and_speaker_metadata_without_required_manifest(tmp_path, layout):
    engine = LLMInference.__new__(LLMInference)
    engine.drama = 'breaking_bad'
    engine.speaker_root = str(tmp_path)
    engine._speaker_cache = {}
    engine._speaker_sources = {}
    engine._speaker_cache_lock = threading.Lock()
    drama = tmp_path / engine.drama
    for name, step, data in [
        ('subtitle_data', 'subtitle_checking_step', {'object_list': {'0': {
            'start_frame': 0, 'end_frame': 20, 'text': 'dialogue'}}}),
        ('speaker_data', 'speaker_processing_step', {'object_list': {'0': {
            'start_frame': 0, 'end_frame': 20, 'text': 'dialogue', 'role': 'Walter'}}}),
    ]:
        parent = drama / name
        directory = parent if layout == 'flat' else parent / 'v1'
        directory.mkdir(parents=True)
        (directory / '0009.json').write_text(json.dumps(data))
        if layout == 'manifest':
            (parent / 'version.json').write_text(json.dumps({step: {'active_version': 'v1'}}))
    subtitles = engine._load_subtitle_json(str(drama), '0009')
    assert engine.lookup_subtitle(subtitles, 10) == 'dialogue'
    intervals, starts, ends = engine._load_speaker_episode('breaking_bad', '0009')
    assert intervals == [(0, 20, 'dialogue', 'Walter')]
    assert starts == [0] and ends == [20]


def test_flat_files_precede_stale_version_and_workspace(tmp_path):
    (tmp_path / '0009.json').write_text('{}')
    (tmp_path / 'v1').mkdir()
    (tmp_path / '.workspace').mkdir()
    (tmp_path / 'version.json').write_text(json.dumps({'step': {'active_version': 'v1'}}))
    assert resolve_data_dir(tmp_path, 'step') == str(tmp_path)


def test_workspace_alone_is_not_an_episode_directory(tmp_path):
    (tmp_path / '.workspace').mkdir()
    assert resolve_data_dir(tmp_path, 'step') is None
