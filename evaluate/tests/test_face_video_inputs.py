"""Face-video sampling has no optional source switch or face-redrawing path."""
import json
from pathlib import Path
from PIL import ImageFont

import numpy as np
import pytest

from dramavqa.cli import build_parser, engine_command
from MultiPointEvalOpen import prepare_frames
from SinglePointEvalOpen import legacy_compatible_runner as sampler


REMOVED_FLAGS = ('--use-rendered-video', '--use_rendered_video')


@pytest.mark.parametrize('task', ['single-point', 'multi-point', 'multi-hop'])
@pytest.mark.parametrize('flag', REMOVED_FLAGS)
def test_public_entry_rejects_removed_source_switch(task, flag, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(['prepare', '--task', task, '--drama', 'sample', flag])
    assert exc.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('entry', ['sampler', 'multi-scene'])
@pytest.mark.parametrize('flag', REMOVED_FLAGS)
def test_internal_entries_reject_removed_source_switch(entry, flag, capsys):
    if entry == 'sampler':
        run = sampler.main
        args = ['--drama', 'sample', '--question_path', 'qa.json', '--model_name', 'test']
    else:
        run = prepare_frames.main
        args = ['--drama', 'sample', '--qa', 'qa.json', '--kind', 'multi_point',
                '--video-root', 'media', '--output-root', 'out']
    with pytest.raises(SystemExit) as exc:
        run(args + [flag])
    assert exc.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('task', ['single-point', 'multi-point', 'multi-hop'])
def test_generated_commands_do_not_pass_removed_switch(task):
    args = build_parser().parse_args(['prepare', '--task', task, '--drama', 'sample',
        '--video-root', 'media', '--output-dir', 'out'])
    args.level = 'L3'
    command = engine_command(args, Path('qa.json'))
    assert not set(REMOVED_FLAGS) & set(command)
    if task != 'single-point':
        nested = prepare_frames.command(prepare_frames.argparse.Namespace(
            python='python', drama='sample', video_root=Path('media'),
            output_root=Path('out'), sample_frames=256, target_width=392), Path('qa.json'))
        assert not set(REMOVED_FLAGS) & set(nested)


def test_raw_video_alone_is_not_an_input_fallback(tmp_path):
    raw = tmp_path / 'media/sample/video_data'
    raw.mkdir(parents=True)
    # The raw episode exists, but must never be opened instead of a face video.
    (raw / '0001.mp4').write_bytes(b'not a face video')
    with pytest.raises(FileNotFoundError, match='Required face video directory not found'):
        sampler.main(['--drama', 'sample', '--question_path', str(tmp_path / 'qa.json'),
            '--model_name', 'fixture', '--serving_mode', 'local', '--only_gen_img',
            '--video_path', str(tmp_path / 'media'), '--image_path', str(tmp_path / 'out')])


def test_save_256_face_frames_without_fonts_or_face_metadata(tmp_path, monkeypatch):
    def no_fonts(*args, **kwargs):
        pytest.fail('Text-subtitle face frames must not load a rendering font')

    monkeypatch.setattr(ImageFont, 'truetype', no_fonts)
    # A stray face metadata file must not be read or used to redraw existing labels.
    face = tmp_path / 'face_data'
    face.mkdir()
    (face / '0001.json').write_text('invalid and deliberately unused')
    subtitle = tmp_path / 'subtitle_data'
    subtitle.mkdir()
    (subtitle / '0001.json').write_text(json.dumps({'object_list': {
        '0': {'start_frame': 0, 'end_frame': 255, 'text': 'dialogue'}}}))
    engine = sampler.LLMInference.__new__(sampler.LLMInference)
    engine.target_width = 0
    engine.scale_factor = 1
    engine._get_drama_root = lambda _episode: str(tmp_path)
    frames = np.full((256, 28, 28, 3), (17, 79, 131), dtype=np.uint8)
    output = tmp_path / 'images/0001-window'
    engine.render([['1_1', frames, list(range(256))]], output.as_posix())
    assert len(list((output / 'render').glob('*.jpg'))) == 256
    assert set(json.loads((output / 'subtitles.json').read_text()).values()) == {'dialogue'}
    assert not hasattr(engine, 'draw_box')
