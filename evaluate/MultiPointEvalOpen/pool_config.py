"""Explicit request and image policy for local pool-compatible evaluation."""
import base64
import io
from PIL import Image


def load_profile(name):
    factors = {'local-qwen': 32, 'local-glm': 28}
    return {'alignment_factor': factors[name], 'max_tokens': 4000,
            'temperature': 0.1, 'top_p': 0.01}


def image_part(image, factor, encoding):
    size = tuple(max(factor, round(v / factor) * factor) for v in image.size)
    image = image.convert('RGB')
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    out = io.BytesIO()
    if encoding == 'png':
        image.save(out, format='PNG')
        mime = 'png'
    else:
        image.save(out, format='WEBP', lossless=True, quality=100, method=3, exact=True)
        with Image.open(io.BytesIO(out.getvalue())) as decoded:
            if decoded.convert('RGB').tobytes() != image.tobytes():
                raise ValueError('Lossless encoding changed RGB pixels')
        mime = 'webp'
    return {'type': 'image_url', 'image_url': {'url':
            f'data:image/{mime};base64,' + base64.b64encode(out.getvalue()).decode('ascii')}}


def request_kwargs(profile, model, messages, max_tokens, thinking):
    result = dict(model=model, messages=messages, max_tokens=max_tokens,
                  temperature=profile['temperature'], top_p=profile['top_p'])
    if thinking != 'omit':
        transport = profile.get('thinking_transport', 'chat-template')
        if transport == 'chat-template':
            result['extra_body'] = {'chat_template_kwargs': {'enable_thinking': thinking == 'on'}}
        elif transport == 'enable_thinking':
            result['extra_body'] = {'enable_thinking': thinking == 'on'}
        elif transport == 'thinking':
            result['extra_body'] = {'thinking': {'type': 'enabled' if thinking == 'on' else 'disabled'}}
        else:
            raise ValueError('Unknown thinking transport')
    return result
