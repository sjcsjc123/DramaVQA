"""Bundled fonts must render Chinese and respect explicit font overrides."""
from pathlib import Path

from PIL import ImageFont
from SinglePointEvalOpen.legacy_compatible_runner import load_render_font
import SinglePointEvalOpen


FONTS = Path(SinglePointEvalOpen.__file__).parent / 'fonts'


def test_default_font_renders_distinct_chinese_characters(monkeypatch, tmp_path):
    monkeypatch.delenv('EVAL_FONT_PATH', raising=False)
    monkeypatch.chdir(tmp_path)
    font = load_render_font(24)
    assert 'SC' in font.getname()[0]
    assert bytes(font.getmask('中')) != bytes(font.getmask('文'))


def test_explicit_english_font_is_used(monkeypatch):
    monkeypatch.setenv('EVAL_FONT_PATH', str(FONTS / 'NotoSans-Regular.ttf'))
    font = load_render_font(24)
    assert font.getname()[0] == 'Noto Sans'
    assert bytes(font.getmask('A')) != bytes(font.getmask('B'))
