import base64
import io
import unittest
from PIL import Image, features
import pytest
from MultiPointEvalOpen.pool_config import image_part, load_profile, request_kwargs

class PoolPolicyTests(unittest.TestCase):
    def test_local_request(self):
        p = load_profile('local-qwen')
        r = request_kwargs(p, 'model', [], 4000, 'off')
        self.assertEqual(r['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual((r['temperature'], r['top_p']), (0.1, 0.01))
        self.assertNotIn('extra_body', request_kwargs(p, 'model', [], 4000, 'omit'))


@pytest.mark.parametrize('encoding', ['png', 'webp-lossless'])
def test_lossless_pixels_and_alignment(encoding):
    if encoding == 'webp-lossless' and not features.check('webp'):
        pytest.skip('Pillow needs WebP codec support')
    image = Image.new('RGB', (392, 224), (13, 71, 129))
    for name, size in [('local-qwen', (384, 224)), ('local-glm', (392, 224))]:
        part = image_part(image, load_profile(name)['alignment_factor'], encoding)
        data = base64.b64decode(part['image_url']['url'].split(',')[1])
        with Image.open(io.BytesIO(data)) as actual:
            assert actual.size == size
            assert actual.convert('RGB').getpixel((0, 0)) == (13, 71, 129)

def test_multihop_policy_forwarding():
    from pathlib import Path
    from dramavqa.cli import build_parser, engine_command
    for task in ('multi-point', 'multi-hop'):
        args = build_parser().parse_args(['run', '--task', task, '--drama', 'demo',
            '--image-root', 'images', '--output-dir', 'out', '--level', 'L3',
            '--model', 'model', '--max-tokens', '4000', '--temperature', '0.1',
            '--evaluation-profile', 'local-glm'])
        cmd = engine_command(args, Path('qa.json'))
        assert cmd[cmd.index('--evaluation-profile') + 1] == 'local-glm'
        assert cmd[cmd.index('--model-thinking') + 1] == 'off'
        assert cmd[cmd.index('--thinking-transport') + 1] == 'chat-template'


def test_legacy_profile_is_rejected():
    import pytest
    from MultiPointEvalOpen.pool_config import load_profile
    with pytest.raises(KeyError):
        load_profile('legacy')
