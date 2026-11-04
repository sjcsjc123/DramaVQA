"""Exercise flat metadata reads and legacy media layout compatibility."""
import json

import pytest

from SinglePointEvalOpen.media_layout import resolve_data_dir


@pytest.mark.parametrize('layout', ['flat', 'manifest', 'subdirectory'])
def test_metadata_directory_resolution(tmp_path, layout):
    parent = tmp_path / 'subtitle_data'
    directory = parent if layout == 'flat' else parent / 'v1'
    directory.mkdir(parents=True)
    (directory / '0009.json').write_text('{}')
    if layout == 'manifest':
        (parent / 'version.json').write_text(json.dumps({
            'subtitle_checking_step': {'active_version': 'v1'}}))
    assert resolve_data_dir(parent, 'subtitle_checking_step') == str(directory)


def test_flat_files_precede_stale_version_and_workspace(tmp_path):
    (tmp_path / '0009.json').write_text('{}')
    (tmp_path / 'v1').mkdir()
    (tmp_path / '.workspace').mkdir()
    (tmp_path / 'version.json').write_text(json.dumps({'step': {'active_version': 'v1'}}))
    assert resolve_data_dir(tmp_path, 'step') == str(tmp_path)


def test_workspace_alone_is_not_an_episode_directory(tmp_path):
    (tmp_path / '.workspace').mkdir()
    assert resolve_data_dir(tmp_path, 'step') is None
