#!/usr/bin/env python3
"""Prepare exp3 offline, or validate/copy selected ablation inputs before vLLM starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DRAMAS = ['the_long_night', 'ode_to_joy_1', 'minning_town', 'qin_empire_2',
          'the_knockout', 'stand_by_me_1', 'three_body', 'battle_of_changsha',
          'friends', 'lost', 'breaking_bad', 'prison_break']


def engine_for(args, drama, level, frames, qa):
    from ask_vllm import LLMInference
    engine = LLMInference(SimpleNamespace(
        port='22002', drama=drama, level=level, sample_frames=frames,
        scale_factor=0.2, target_width=args.width, subtitle_text=True, render_subtitle=False,
        ablation_mode='exp3_gallery', image_path=str(ROOT / 'unused-images'),
        model_name='qwen3-vl-8b', model_path='', question_path=str(qa), video_path=args.video_root,
        use_render_info=True, use_rendered_video=False, parallel=False, num_workers=1,
        group_4x4=False, group_2x2=False, media_resolution='low', only_gen_img=False,
        result_path='', serving_mode='local'))
    return engine


def times_for(qa, level):
    data = json.loads(qa.read_text())
    if any('Q&A' in clip for clip in data.values() if isinstance(clip, dict)):
        clips = data.values()
    else:
        clips = (clip for category in ('low-level', 'high-level', 'multi-point')
                 for clip in data.get(category, {}).values())
    return sorted({clip[level + '_time'] for clip in clips if clip.get(level + '_time')})


def copy_checked(src, dst):
    """Atomic file publication; hashes avoid trusting old size-only completion markers."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = lambda p: hashlib.sha256(p.read_bytes()).digest()
    if dst.is_file() and src.stat().st_size == dst.stat().st_size and digest(src) == digest(dst):
        return
    fd, tmp = tempfile.mkstemp(prefix='.copy-', dir=dst.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp)
        if digest(src) != digest(Path(tmp)):
            raise IOError('Copy verification failed: ' + str(src))
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare-gallery', 'check', 'stage'])
    parser.add_argument('--exp', choices=['exp1_burn', 'exp3_gallery', 'exp4_speaker', 'exp5_nosub'], default=os.environ.get('ABLATION_MODE', 'exp3_gallery'))
    parser.add_argument('--qa-root', default=os.environ.get('QA_DATA_PATH'))
    parser.add_argument('--video-root', default=os.environ.get('VIDEO_ROOT'))
    parser.add_argument('--image-root', default=os.environ.get('IMAGE_ROOT', str(Path.cwd() / 'images')))
    parser.add_argument('--gallery-root', default=os.environ.get('GALLERY_CACHE_ROOT', str(Path.cwd() / 'gallery_cache')))
    parser.add_argument('--dst', default=os.environ.get('LOCAL_IMAGES_DST', str(Path.cwd() / 'staged_images')))
    parser.add_argument('--levels', default=os.environ.get('LEVELS', 'L3'))
    parser.add_argument('--frames', default=os.environ.get('SAMPLE_FRAMES_LIST', '256'))
    parser.add_argument('--width', type=int, default=int(os.environ.get('TARGET_WIDTH', '392')))
    parser.add_argument('--dramas', default=' '.join(DRAMAS))
    parser.add_argument('--limit', type=int, default=0, help='Preparation/check only; never stage a partial inference set')
    args = parser.parse_args()
    if not args.qa_root or not args.video_root or args.width <= 0:
        parser.error('--qa-root, --video-root and positive --width are required')
    if args.action == 'prepare-gallery' and args.exp != 'exp3_gallery':
        parser.error('prepare-gallery requires exp3_gallery')
    if args.action == 'stage' and args.limit:
        parser.error('stage does not permit --limit')
    source = Path(args.gallery_root if args.exp == 'exp3_gallery' else args.image_root)
    destination = Path(args.dst) / args.exp
    if args.action == 'stage' and (destination.resolve() == source.resolve() or source.resolve() in destination.resolve().parents):
        parser.error('Destination must be separate from shared source')
    plans = []
    count = 0
    for level in args.levels.replace(',', ' ').split():
        for frames in map(int, args.frames.replace(',', ' ').split()):
            for drama in args.dramas.replace(',', ' ').split():
                filename = drama
                qa = Path(args.qa_root) / (filename + '.json')
                if not qa.is_file():
                    raise FileNotFoundError('Missing QA (refusing partial coverage): ' + str(qa))
                times = times_for(qa, level)
                if args.limit:
                    times = times[:args.limit]
                engine = engine_for(args, drama, level, frames, qa) if args.exp == 'exp3_gallery' else None
                for time_str in times:
                    if engine:
                        from gallery_cache import cache_path, identity, load, save
                        path = cache_path(source, engine, time_str)
                        expected = identity(engine, time_str)
                        try:
                            load(path, expected, images=False)
                        except Exception as exc:
                            if args.action != 'prepare-gallery':
                                raise RuntimeError('Missing/stale exp3 cache: %s. Run prepare_local_inputs.py prepare-gallery with identical QA/video/levels/frames/width before submitting.' % path) from exc
                            save(path, expected, engine._extract_gallery_clip(time_str))
                        plans.append((path, destination / path.relative_to(source)))
                    else:
                        suffix = 'sub' if args.exp == 'exp1_burn' else 'nosub'
                        clip = source / level / str(frames) / ('results.w%d_%s' % (args.width, suffix)) / 'img_data' / drama / time_str.replace(':', '-').replace('#', '-')
                        images = sorted((clip / 'render').glob('*.jpg'))
                        if len(images) != frames:
                            raise ValueError('Missing/incomplete %s images (%d/%d): %s. Prepare the correct subtitle variant first; nosub is not burn.' % (args.exp, len(images), frames, clip))
                        files = list(images)
                        if args.exp == 'exp4_speaker':
                            subs = clip / 'subtitles.json'
                            mapping = json.loads(subs.read_text())
                            if set(p.name for p in images) - mapping.keys():
                                raise ValueError('Incomplete subtitles: ' + str(subs))
                            files.append(subs)
                        plans.extend((p, destination / p.relative_to(source)) for p in files)
                    count += 1
    if not count:
        raise ValueError('No QA windows selected')
    if args.action == 'stage':
        destination.mkdir(parents=True, exist_ok=True)
        needed = sum(src.stat().st_size for src, dst in plans if not dst.exists())
        if shutil.disk_usage(destination).free < needed:
            raise OSError('Insufficient local space: need at least %d bytes' % needed)
        for src, dst in plans:
            copy_checked(src, dst)
    print('[ablation-inputs] %s: %d windows, %d files; destination=%s' % (args.action, count, len(plans), destination))


if __name__ == '__main__':
    main()
