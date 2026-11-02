"""Lossless exp3 cache. Never substitutes rendered face-box frames for raw video."""
import hashlib
import json
import os
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

SCHEMA = 1


def identity(engine, time_str):
    sources = []
    episodes, _, _ = engine._sorted_episodes_in_l3(time_str)
    if not episodes:
        raise RuntimeError('No source episodes for gallery cache: ' + time_str)
    for key, _, ep in episodes:
        video = Path(engine.episode_info[key]['video_path'])
        root = Path(engine._get_drama_root(key))
        files = [video]
        for kind in ('video_data', 'face_data', 'subtitle_data', 'speaker_data'):
            base = root / kind
            files.extend(base.glob('version.json'))
            if kind != 'video_data':
                files.extend(base.glob('*/%04d.json' % ep))
                files.extend(base.glob('*/.workspace/new_roles_map.json'))
        for path in sorted(set(files)):
            stat = path.stat()
            sources.append([root.name + '/' + str(path.relative_to(root)), stat.st_size,
                            stat.st_mtime_ns,
                            hashlib.sha256(path.read_bytes()).hexdigest() if path.suffix == '.json' else None])
    return dict(schema=SCHEMA, kind='raw-video-gallery-text', drama=engine.drama,
                level=engine.level, frames=engine.sample_frames, width=engine.target_width,
                scale=engine.scale_factor if not engine.target_width else None, time=time_str,
                code=hashlib.sha256(Path(__file__).with_name('ask_vllm.py').read_bytes()).hexdigest(),
                sources=sources)


def cache_path(root, engine, time_str):
    key = hashlib.sha256(time_str.encode()).hexdigest()
    return Path(root) / engine.level / str(engine.sample_frames) / ('w%d' % engine.target_width) / engine.drama / (key + '.zip')


def load(path, expected, images=True):
    from PIL import Image
    with zipfile.ZipFile(path) as archive:
        meta = json.loads(archive.read('manifest.json'))
        if meta['identity'] != expected:
            raise ValueError('Stale gallery cache: ' + str(path))
        if len(meta['frames']) != expected['frames'] or len(meta['frame_subs']) != expected['frames']:
            raise ValueError('Incomplete gallery cache: ' + str(path))
        decoded = {}
        for name, digest in meta['hashes'].items():
            blob = archive.read(name)
            if hashlib.sha256(blob).hexdigest() != digest:
                raise ValueError('Corrupt gallery cache: ' + str(path))
            if images:
                with Image.open(BytesIO(blob)) as image:
                    decoded[name] = image.copy()
        required = meta['frames'] + [item['file'] for item in meta['gallery']]
        if set(required) != set(meta['hashes']):
            raise ValueError('Invalid gallery image manifest')
        if images:
            return dict(frames=[decoded[n] for n in meta['frames']], frame_subs=meta['frame_subs'],
                        gallery=[dict(name=x['name'], image=decoded[x['file']]) for x in meta['gallery']])
        return meta


def save(path, expected, clip):
    path = Path(path)
    if len(clip['frames']) != expected['frames'] or len(clip['frame_subs']) != expected['frames']:
        raise ValueError('Refusing incomplete extracted clip')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.gallery-', dir=path.parent)
    os.close(fd)
    try:
        meta = dict(identity=expected, frames=[], gallery=[], frame_subs=clip['frame_subs'], hashes={})
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for kind, items in [('frames', clip['frames']), ('gallery', clip['gallery'])]:
                for index, item in enumerate(items):
                    name = '%s/%06d.png' % (kind, index)
                    image = item if kind == 'frames' else item['image']
                    buffer = BytesIO()
                    image.save(buffer, format='PNG')
                    blob = buffer.getvalue()
                    archive.writestr(name, blob)
                    meta['hashes'][name] = hashlib.sha256(blob).hexdigest()
                    meta[kind].append(name if kind == 'frames' else dict(name=item['name'], file=name))
            archive.writestr('manifest.json', json.dumps(meta, ensure_ascii=False))
        load(temporary, expected, images=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
