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
