import base64
import io
import unittest
from PIL import Image
from MultiPointEvalOpen.pool_config import image_part, load_profile, request_kwargs

class PoolPolicyTests(unittest.TestCase):
    def test_local_request(self):
        p = load_profile('local-qwen')
        r = request_kwargs(p, 'model', [], 4000, 'off')
        self.assertEqual(r['extra_body'], {'chat_template_kwargs': {'enable_thinking': False}})
        self.assertEqual((r['temperature'], r['top_p']), (0.1, 0.01))
        self.assertNotIn('extra_body', request_kwargs(p, 'model', [], 4000, 'omit'))

    def test_lossless_pixels_and_alignment(self):
        image = Image.new('RGB', (392, 224), (13, 71, 129))
        for name, size in [('local-qwen', (384, 224)), ('local-glm', (392, 224))]:
            for encoding in ('png', 'webp-lossless'):
                part = image_part(image, load_profile(name)['alignment_factor'], encoding)
                data = base64.b64decode(part['image_url']['url'].split(',')[1])
                with Image.open(io.BytesIO(data)) as actual:
                    self.assertEqual(actual.size, size)
                    self.assertEqual(actual.convert('RGB').getpixel((0, 0)), (13, 71, 129))

def test_multihop_policy_forwarding():
    from types import SimpleNamespace
    from pathlib import Path
    from dramavqa.cli import engine_command
    for task in ('multi-point', 'multi-hop'):
        args = SimpleNamespace(task=task, command='run', drama='demo', image_root=Path('images'),
            output_dir=Path('out'), level='L3', sample_frames=8, target_width=392,
            model='model', base_url='http://localhost/v1', num_workers=1, max_tokens=4000,
            api_timeout=120, evaluation_profile='local-glm', model_thinking='off',
            image_encoding='png', thinking_transport='chat-template')
        cmd = engine_command(args, Path('qa.json'))
        assert cmd[cmd.index('--evaluation-profile') + 1] == 'local-glm'
        assert cmd[cmd.index('--model-thinking') + 1] == 'off'
        assert cmd[cmd.index('--thinking-transport') + 1] == 'chat-template'


def test_legacy_profile_is_rejected():
    import pytest
    from MultiPointEvalOpen.pool_config import load_profile
    with pytest.raises(KeyError):
        load_profile('legacy')
