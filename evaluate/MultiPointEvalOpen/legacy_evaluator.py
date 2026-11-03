"""Shared multi-scene evaluation engine for multi-point and multi-hop tasks.

Loads two or three L3 windows, orders scenes chronologically, and sends their
frames and text subtitles together. Public callers should use dramavqa.
Historical answer extraction, fallback and scoring are preserved.
"""

import io
import os
import re
import sys
import json
import time
import base64
import random
import argparse
import threading
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image


# ---------------------------------------------------------------------------
# prompt (full + nosub, consistent with ask_vllm.py SYS_PROMPT_MODE=full +
# subtitle_text)
# ---------------------------------------------------------------------------

_SYSTEM_HEAD = (
    "These are the frames of a video. The subtitle line for each frame is "
    "provided as text right after that frame.\n"
    "Select the best answer to the following single-choice question based on the video."
)

_SYSTEM_FORMAT = (
    "\nRespond with only the letter of the correct option, enclosed in angle brackets, "
    "e.g. <letter>."
)

_SYSTEM_EXAMPLE = (
    "\n\nExample:\n"
    "Question: In the bedroom scene, what is the main color of Hillary's shirt?\n"
    "A. White\nB. Black\nC. Green\nD. Yellow\nE. Blue\nF. Purple\n"
    "Answer: <letter>"
)

SYSTEM_PROMPT = _SYSTEM_HEAD + _SYSTEM_FORMAT + _SYSTEM_EXAMPLE

QA_TYPES = ['B_order', 'B_sort', 'D_same', 'multi_hop']

# scene delimiter tag (multi-scenario questions)
SCENE_LABEL_TMPL = "=== Scene {n} ==="

# Answer-extraction patterns are ordered by priority and shared with
# tools/reextract_answers.py. Match explicit final-answer markers before permissive
# fallbacks, so option text such as "A. white" is not mistaken for the model choice.
_ANSWER_PATTERNS = [
    (r'<\|begin_of_box\|>\s*([A-Za-z])\s*<\|end_of_box\|>', 0),
    (r'(?:Final\s*Answer|Answer|答案)\s*[:：]\s*\**\s*\[?<?\s*([A-Za-z])\s*>?\]?\s*[.、,，)]?\s*\**\s*$',
     re.IGNORECASE),
    (r'Final\s*Answer\s*[:：]\s*<?\s*([A-Za-z])\s*>?', re.IGNORECASE),
    (r'<\s*(?:letter|answer|option|choice)\s*>\s*\**\s*([A-Za-z])\s*\**\s*<\s*/',
     re.IGNORECASE),
    (r'<\s*([A-Za-z])\s*>', 0),
    (r'\[\s*([A-Za-z])\s*\]', 0),
    (r'(?:Answer|答案)\s*[:：]\s*<?\s*([A-Za-z])\s*>?[.、,，)\s]', re.IGNORECASE),
    # custom tag wrapping such as <ref>B</ref> / <ans>B</ans>
    (r'<\s*(?:ref|ans)\s*>\s*([A-Za-z])\s*<\s*/', re.IGNORECASE),
    # Opening tag truncated at the end, with the character already emitted but the
    # closing symbol not yet arrived: '...\n\n<C'
    (r'<\s*([A-Za-z])\s*$', 0),
]
_ANSWER_FULLMATCH = r'\**\s*([A-Za-z])\s*[.、,，)]?\s*\**'
# Fallback: accept a final nonempty line containing only an option letter. Qwen3.6-27B
# can end with a bare C rather than angle brackets; otherwise extraction fails and
# random guessing follows. On 6,343 previously extracted responses, this matched 322
# earlier results with zero conflicts.
_TAIL_LETTER_RE = re.compile(r'([A-Za-z])\s*[.、,，)]?')
# Second fallback: accept a final line that copies the full option, such as "A. The
# former happens earlier". Require the letter at line start followed by a period, space,
# and text, so ordinary prose is not mistaken for an answer.
_TAIL_OPTION_RE = re.compile(r'^([A-Za-z])\.\s+\S')
_OPTION_LINE_RE = re.compile(r'^\s*\**\s*([A-Z])\s*[.、,，):：]', re.MULTILINE)

# Qwen-VL requires width and height aligned to multiples of 28.
QWEN_VL_SIZE_FACTOR = 28


# ---------------------------------------------------------------------------
# utility function
# ---------------------------------------------------------------------------

def extract_answer_letter(text):
    'Extract option letters from model response, return None if not found.'
    if not text:
        return None
    stripped = text.strip()
    for pattern, flags in _ANSWER_PATTERNS:
        m = re.search(pattern, stripped, flags) if flags else re.search(pattern, stripped)
        if m:
            return m.group(1)
    m = re.fullmatch(_ANSWER_FULLMATCH, stripped)
    if m:
        return m.group(1)
    # Last-line fallback: models often end with bare letters "...C" or "C. Option text",
    # which above rules cannot capture
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].replace('*', '').replace('#', '').strip()
        m = _TAIL_LETTER_RE.fullmatch(last)
        if m:
            return m.group(1)
        m = _TAIL_OPTION_RE.match(last)
        if m:
            return m.group(1)
    return None


def parse_option_letters(options_str):
    'Parse the alphabet set of options for random fallback usage.'
    if not isinstance(options_str, str):
        return []
    letters = []
    for m in _OPTION_LINE_RE.finditer(options_str):
        ch = m.group(1).upper()
        if ch not in letters:
            letters.append(ch)
    return letters


def l3_time_to_dir_name(l3_time_str):
    """L3_time field -> image cache directory name. Supports two input formats: Format A
    (with S3# prefix): 'S3#0002-00:12:18.071_S3#0002-00:21:03.971' ->
    'S3-0002-00-12-18.071_S3-0002-00-21-03.971' Format B (without prefix):
    '0011-00:22:01.520_0011-00:29:03.440' -> '0011-00-22-01.520_0011-00-29-03.440'
    Rules: '#' -> '-', ':' -> '-', otherwise unchanged. Consistent with
    time_.replace(':','-').replace('#','-') in ask_vllm.py."""
    return l3_time_str.replace('#', '-').replace(':', '-')


# L3_time window start: optional season number + episode number + time:minute:second
_L3_START_RE = re.compile(
    r'(?:S(\d+)#)?(\d+)-(\d+):(\d+):(\d+(?:\.\d+)?)'
)


def parse_l3_start(l3_time_str):
    """Parse the start time of the L3_time window within the story, used for sorting scenes
    by story time. Returns (season, episode, start_seconds). If parsing fails, returns a
    large sentinel value to ensure unrecognizable segments are placed last without
    disrupting the relative order of other segments."""
    start = (l3_time_str or '').split('_')[0]
    m = _L3_START_RE.search(start)
    if not m:
        return (10 ** 9, 10 ** 9, 10 ** 9.0)
    season = int(m.group(1) or 0)
    episode = int(m.group(2))
    hh, mm, ss = int(m.group(3)), int(m.group(4)), float(m.group(5))
    return (season, episode, hh * 3600 + mm * 60 + ss)


def sort_scenes_by_story_time(l3_times, scenes):
    """Sort scenes by their real-time order in the story. scenes and l3_times are of equal
    length, each element is (imgs, subs). Returns list[(orig_idx, imgs, subs)], where
    orig_idx is the original L3_time / question scene index (0-based)."""
    paired = []
    for i, (t, scene) in enumerate(zip(l3_times, scenes)):
        paired.append((parse_l3_start(t), i, scene[0], scene[1]))
    paired.sort(key=lambda x: (x[0], x[1]))
    return [(orig_idx, imgs, subs) for _, orig_idx, imgs, subs in paired]


# drama name in QA JSON key may differ from image cache directory name; unified mapping
# here.
_DRAMA_DIR_MAP = {
    'qin_empire_2': 'qin_empire_2',
}


def drama_to_image_dir(drama):
    'Map QA JSON drama name to image cache directory name (if different).'
    return _DRAMA_DIR_MAP.get(drama, drama)


def resolve_drama_image_dir(image_root, drama):
    """Prefer the public sampler's directory, with legacy aliases as fallback."""
    if os.path.isdir(os.path.join(image_root, drama)):
        return drama
    return drama_to_image_dir(drama)


def align_to_factor(value, factor=QWEN_VL_SIZE_FACTOR):
    return max(factor, int(round(value / factor)) * factor)


def align_pil(pil_img):
    'Align width and height to multiples of 28 (required for Qwen-VL token matching).'
    w, h = pil_img.size
    new_w = align_to_factor(w)
    new_h = align_to_factor(h)
    if (new_w, new_h) != (w, h):
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return pil_img


def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def load_clip_images(image_root, drama, l3_time):
    """Read all frame images from <image_root>/<drama_dir>/<clip_dir>/render/. drama name
    uses the QA filename stem for the media and image directory. Returns: imgs: list[(PIL.Image,
    str)] — (image, filename), sorted by filename reason: str | None — None=success,
    otherwise skip_reason"""
    clip_dir = os.path.join(image_root, resolve_drama_image_dir(image_root, drama), l3_time_to_dir_name(l3_time), 'render')
    if not os.path.isdir(clip_dir):
        return None, 'images_not_found'

    files = sorted(
        [f for f in os.listdir(clip_dir)
         if not f.startswith('.') and f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: tuple(int(p) for p in x.split('.')[0].split('_'))
    )
    if not files:
        return None, 'images_empty'

    imgs = []
    for fname in files:
        try:
            img = Image.open(os.path.join(clip_dir, fname))
            img.load()
            imgs.append((img, fname))
        except Exception as e:
            print('[WARN] 读取图片失败，跳过: {}/{}: {}'.format(clip_dir, fname, e))
    if not imgs:
        return None, 'images_load_failed'
    return imgs, None


def load_subtitles(image_root, drama, l3_time):
    'Read <clip>/subtitles.json (nosub mode generated by gen_images).'
    sub_path = os.path.join(
        image_root, resolve_drama_image_dir(image_root, drama), l3_time_to_dir_name(l3_time), 'subtitles.json')
    try:
        with open(sub_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def atomic_write_jsonl(path, record, lock):
    'Thread-safe append a JSON record to a .jsonl file.'
    line = json.dumps(record, ensure_ascii=False) + '\n'
    with lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)


def atomic_dump_json(obj, path):
    'Atomic write JSON: write to temporary file -> os.replace.'
    tmp = '{}.tmp.{}.{}'.format(path, os.getpid(), threading.get_ident())
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# TPM token bucket (reuses ask_vllm.py design, does not rely on environment variables)
# ---------------------------------------------------------------------------

class TokenBudget:
    def __init__(self, tpm):
        self.enabled = tpm > 0
        if not self.enabled:
            return
        self.capacity = tpm * 0.85
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, need):
        if not self.enabled:
            return
        need = min(need, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= need:
                    self.tokens -= need
                    return
                wait = (need - self.tokens) / self.rate
            time.sleep(min(wait, 30) + random.uniform(0, 0.5))


def estimate_tokens(n_images, text_chars, max_tokens):
    'Rough estimate of request token count (consistent with ask_vllm.py).'
    return n_images * 85 + text_chars // 4 + max_tokens


# ---------------------------------------------------------------------------
# inference engine
# ---------------------------------------------------------------------------

def load_keys(api_key, keys_file):
    """Return a list of keys. Prioritize --keys-file, then --api-key. keys_file format: one
    key per line, lines starting with # are comments."""
    if keys_file:
        keys = []
        with open(keys_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keys.append(line)
        if not keys:
            raise ValueError('--keys-file {} 中没有有效 key'.format(keys_file))
        return keys
    if api_key:
        return [api_key]
    raise ValueError('必须提供 --api-key 或 --keys-file')


class MultiPointEvaluator:
    """Drama_Multi_Point_QA evaluation engine. All configurations are passed as constructor
    parameters and do not read any environment variables. Supports multi-key
    round-robin: each API call rotates to the next key."""

    def __init__(self, base_url, api_key, model, image_root, output_dir,
                 num_workers=2, tpm_limit=0, max_tokens=1000,
                 max_retries=6, api_timeout=120.0, limit=0,
                 keys_file='', input_fingerprints=None, reasoning_effort=None):
        self.model = model
        self.image_root = image_root
        self.input_fingerprints = input_fingerprints
        self.reasoning_effort = reasoning_effort
        self._routed_block = threading.Event()
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.limit = limit

        # multi-key round-robin
        self._keys = load_keys(api_key, keys_file)
        self._key_idx = 0
        self._key_lock = threading.Lock()
        self._base_url = base_url
        self._api_timeout = api_timeout

        self.budget = TokenBudget(tpm_limit)
        self._raw_log_lock = threading.Lock()
        print('  API key 数量: {}'.format(len(self._keys)))

    def _get_client(self):
        'Each call rotates to the next key, returning a new OpenAI client.'
        from openai import OpenAI, DefaultHttpxClient
        with self._key_lock:
            key = self._keys[self._key_idx % len(self._keys)]
            self._key_idx += 1
        return OpenAI(
            api_key=key,
            base_url=self._base_url,
            http_client=DefaultHttpxClient(verify=self.input_fingerprints is not None),
            timeout=self._api_timeout,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # message construction
    # ------------------------------------------------------------------

    def _build_messages(self, question, options, scenes):
        """Build multi-scenario messages. scenes: list[(orig_idx, imgs_with_names,
        subtitles)] orig_idx: original L3_time index (0-based), corresponding to Scene N
        in the question imgs_with_names: list[(PIL.Image, str)] subtitles: dict[filename
        -> text] Segments are concatenated in chronological order (earliest first). "===
        Scene N ===" still corresponds to the original Scene N / former-latter, not the
        display order. Each frame image is followed by its corresponding subtitle text
        (nosub mode)."""
        user_content = [
            {
                'type': 'text',
                'text': (
                    'The video clips below are concatenated in chronological '
                    'storyline order (earliest first). The "=== Scene N ===" '
                    'label matches Scene N in the question, not the display order.\n'
                    'Question: {}\n{}'.format(question, options)
                ),
            },
        ]

        for orig_idx, imgs_with_names, subs in scenes:
            user_content.append({
                'type': 'text',
                'text': SCENE_LABEL_TMPL.format(n=orig_idx + 1),
            })
            for pil_img, fname in imgs_with_names:
                pil_img = align_pil(pil_img)
                b64 = pil_to_b64(pil_img)
                user_content.append({
                    'type': 'image_url',
                    'image_url': {'url': 'data:image/jpeg;base64,' + b64},
                })
                if subs:
                    sub_text = (subs.get(fname) or '').strip()
                    if sub_text:
                        user_content.append({'type': 'text', 'text': sub_text})

        return [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': user_content},
        ]

    # ------------------------------------------------------------------
    # API call (exponential backoff retry, consistent with ask_vllm.py openai_inference
    # logic)
    # ------------------------------------------------------------------

    def _call_routed_api(self, messages):
        """Source-aware continuation: bounded retry, redacted errors, quota stop."""
        n_images = sum(p.get('type') == 'image_url' for p in messages[1]['content'])
        chars = sum(len(p.get('text', '')) for p in messages[1]['content'])
        self.budget.acquire(estimate_tokens(n_images, chars, self.max_tokens))
        kwargs = dict(model=self.model, messages=messages, max_tokens=self.max_tokens,
                      temperature=0.1, top_p=0.01)
        is_vl = 'qwen' in self.model.lower() and '-vl-' in self.model.lower()
        if not is_vl:
            kwargs['extra_body'] = {'enable_thinking': False}
        for attempt in range(self.max_retries):
            if self._routed_block.is_set():
                return None, 0, 'provider_blocked'
            try:
                with self._get_client() as client:
                    response = client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                raw = (message.content or '').strip()
                if not is_vl and (getattr(message, 'reasoning_content', '') or '').strip():
                    self._routed_block.set()
                    return None, 0, 'thinking_not_disabled'
                return raw, response.usage.total_tokens if response.usage else 0, response.choices[0].finish_reason
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                quota = any(word in str(exc).lower() for word in ('insufficient_quota', 'insufficient_balance', 'credit', 'quota'))
                if quota or (status is not None and status < 500 and status != 429):
                    self._routed_block.set()
                    return None, 0, 'provider_blocked'
                if attempt == self.max_retries - 1:
                    return None, 0, 'api_no_response'
                time.sleep(min(2 ** attempt, 30))
        return None, 0, 'api_no_response'

    def _call_api(self, messages, qa_key):
        if self.input_fingerprints is not None:
            return self._call_routed_api(messages)
        import openai as _openai

        n_imgs = sum(
            1 for part in messages[1]['content']
            if isinstance(part, dict) and part.get('type') == 'image_url'
        )
        text_chars = sum(
            len(part.get('text', ''))
            for part in messages[1]['content']
            if isinstance(part, dict) and part.get('type') == 'text'
        )
        self.budget.acquire(estimate_tokens(n_imgs, text_chars, self.max_tokens))

        for attempt in range(self.max_retries):
            try:
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print('{} [{}] model={} attempt={}/{} imgs={}'.format(
                    ts, qa_key, self.model, attempt + 1, self.max_retries, n_imgs))

                # Follow ask_vllm.py for enable_thinking: omit it for Qwen3-VL because
                # the siliconflow endpoint rejects it with HTTP 400. For other models
                # such as Qwen3.6-27B, send enable_thinking=False in top-level
                # extra_body; this suppresses reasoning_content and reduces completion
                # tokens.
                model_lower = self.model.lower()
                is_qwen_vl = 'qwen' in model_lower and '-vl-' in model_lower
                is_glm53 = 'glm-5.3' in model_lower or 'glm53' in model_lower.replace('.', '').replace('-', '')
                create_kwargs = dict(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=0.1,
                    top_p=0.01,
                )
                if is_glm53:
                    # Native GLM-5.3 template always thinks; enable_thinking=off is invalid.
                    if self.reasoning_effort:
                        create_kwargs['reasoning_effort'] = self.reasoning_effort
                elif not is_qwen_vl:
                    create_kwargs['extra_body'] = {'enable_thinking': False}
                resp = self._get_client().chat.completions.create(**create_kwargs)

                raw = (resp.choices[0].message.content or '').strip()
                # Compatible with models that output reasoning_content exclusively
                # (e.g., Qwen3.5, Qwen3.6-27B)
                if not raw:
                    rc = getattr(resp.choices[0].message, 'reasoning_content', None)
                    if (rc or '').strip():
                        raw = rc.strip()

                tokens = resp.usage.total_tokens if resp.usage else 0
                finish = getattr(resp.choices[0], 'finish_reason', None)
                return raw, tokens, finish

            except _openai.RateLimitError as e:
                wait = min(15 * (2 ** attempt), 240) + random.uniform(0, 5)
                print('[{}] RateLimitError，等待 {:.0f}s: {}'.format(
                    qa_key, wait, str(e)[:120]))
                time.sleep(wait)

            except _openai.APITimeoutError as e:
                wait = min(30 * (2 ** attempt), 120) + random.uniform(0, 5)
                print('[{}] Timeout，等待 {:.0f}s'.format(qa_key, wait))
                time.sleep(wait)

            except _openai.OpenAIError as e:
                status = getattr(e, 'status_code', None)
                retryable = status is None or status >= 500
                err_msg = 'OpenAIError status={}: {}'.format(status, str(e)[:240])
                print('[{}] {}'.format(qa_key, err_msg))
                if not retryable or attempt == self.max_retries - 1:
                    return None, 0, 'api_openai_error|' + err_msg
                wait = min(10 * (2 ** attempt), 120) + random.uniform(0, 5)
                time.sleep(wait)

            except Exception as e:
                err_msg = '未预期异常 {}: {}'.format(type(e).__name__, str(e)[:240])
                print('[{}] {}'.format(qa_key, err_msg))
                if attempt == self.max_retries - 1:
                    return None, 0, 'api_unexpected_error|' + err_msg
                time.sleep(15 + random.uniform(0, 3))

        return None, 0, 'api_max_retries'

    # ------------------------------------------------------------------
    # Single-item evaluation
    # ------------------------------------------------------------------

    def _eval_one(self, drama, qa_type, qa_id, qa_item, raw_log_path):
        """Evaluate a single question, returning a result dict. qa_item comes from
        Drama_Multi_Point_QA JSON, with key fields: Question, Options, Answer, L3_time
        (list[str])"""
        key = '{}:{}:{}'.format(drama, qa_type, qa_id)
        l3_times = qa_item.get('L3_time', [])
        question  = qa_item.get('Question', '')
        options   = qa_item.get('Options', '')
        answer    = qa_item.get('Answer', '')

        # Routed mode picks the entire question's namespace; never per-scene fallback.
        image_root = self.image_root
        provenance = {}
        if self.input_fingerprints is not None:
            from pool_inputs import question_image_root, source_groups
            source_groups({drama: {qa_type + '/' + str(qa_id): qa_item}})
            image_root = str(question_image_root(self.image_root, qa_item))
            provenance = {'input_source': qa_item['_input_source'],
                          'input_sha256': self.input_fingerprints[(qa_type, str(qa_id))],
                          'option_letters': parse_option_letters(options)}
            from pool_inputs import inspect_images, question_fingerprint, Blocked
            check = inspect_images(image_root, {drama: {qa_type + '/' + str(qa_id): qa_item}}, 256, True)
            if check['errors'] or question_fingerprint(drama, qa_item, check) != provenance['input_sha256']:
                raise Blocked('Question inputs changed since routed preflight; no request sent')
        # Load frame images for each scene
        scenes = []
        for t in l3_times:
            imgs, reason = load_clip_images(image_root, drama, t)
            if imgs is None:
                print('[WARN] {} L3_time={} 图片缺失({}), 跳过此题'.format(
                    key, t, reason))
                return dict(qa_type=qa_type, qa_id=qa_id, key=key,
                            question=question, options=options, answer=answer,
                            pred=None, result=None, correct=False,
                            total_tokens=0, raw_response=None,
                            skip_reason=reason, error_msg='图片缺失 L3_time={} reason={}'.format(t, reason),
                            guessed=False,
                            ts=datetime.datetime.now().isoformat(), **provenance)
            subs = load_subtitles(image_root, drama, t)
            scenes.append((imgs, subs))

        # Concatenate scenes in chronological order according to real-time in the story;
        # Scene N label still corresponds to the original L3_time index
        orig_order = list(range(len(scenes)))
        scenes = sort_scenes_by_story_time(l3_times, scenes)
        chrono_order = [s[0] for s in scenes]
        if chrono_order != orig_order:
            print('[{}] concat chrono orig={} -> {}'.format(
                key, orig_order, chrono_order))

        # Construct and send message
        messages = self._build_messages(question, options, scenes)
        raw, tokens, finish = self._call_api(messages, key)

        if raw is None:
            # API-level failure; mark skip_reason to indicate failure for next run in
            # format: 'api_openai_error|<details>' or other fixed tag
            skip_tag = (finish or 'api_no_response').split('|')[0]
            err_detail = finish or 'api_no_response'
            return dict(qa_type=qa_type, qa_id=qa_id, key=key,
                        question=question, options=options, answer=answer,
                        pred=None, result=None, correct=False,
                        total_tokens=tokens, raw_response=None,
                        skip_reason=skip_tag, error_msg=err_detail,
                        guessed=False,
                        ts=datetime.datetime.now().isoformat(), **provenance)

        # Extract answer letters
        pred = extract_answer_letter(raw)

        # Save original response to disk
        atomic_write_jsonl(raw_log_path, dict(
            qa_type=qa_type, qa_id=qa_id, key=key,
            raw=raw, pred=pred, answer=answer,
            total_tokens=tokens, finish_reason=finish,
            concat_order='chrono',
            scene_order=chrono_order,
            ts=datetime.datetime.now().isoformat(),
            **provenance,
        ), self._raw_log_lock)

        # Routed continuation never substitutes a random answer.
        if self.input_fingerprints is not None and (pred not in parse_option_letters(options) or finish != 'stop'):
            return dict(qa_type=qa_type, qa_id=qa_id, pred=None, result=None, correct=False,
                        answer=answer, raw_response=raw, total_tokens=tokens, guessed=False,
                        finish_reason=finish, skip_reason='answer_not_extracted' if finish == 'stop' else 'incomplete_response',
                        **provenance)
        # Fallback randomly when extraction fails (retains old interface semantics)
        guessed = False
        if pred is None:
            letters = parse_option_letters(options)
            if letters:
                pred = random.choice(letters)
                guessed = True
                print('[{}] 提取失败，随机兜底 -> {} (候选 {})'.format(
                    key, pred, ''.join(letters)))
            else:
                print('[{}] 提取失败且无法解析选项，记为 None'.format(key))

        correct = (pred == answer) if pred is not None else False
        return dict(qa_type=qa_type, qa_id=qa_id,
                    question=question, answer=answer,
                    pred=pred, result=pred, correct=correct,
                    total_tokens=tokens, raw_response=raw,
                    skip_reason=None, guessed=guessed,
                    finish_reason=finish)

    # ------------------------------------------------------------------
    # Resume from checkpoint: read completed set
    # ------------------------------------------------------------------

    def _load_done_set(self, progress_path):
        """Read the completed (qa_type, qa_id) set from raw_responses.jsonl. Duplicate
        entries for the same question may occur (due to concurrent processes). The last
        entry takes precedence: Only the last entry without skip_reason is considered
        truly completed; if the last entry is skip, the task will be retried next time."""
        last = {}
        if not os.path.isfile(progress_path):
            return set()
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = (obj.get('qa_type', ''), str(obj.get('qa_id', '')))
                    last[k] = obj
        except Exception as e:
            print('[WARN] 读取进度文件失败，从头开始: {}'.format(e))
            return set()
        if self.input_fingerprints is not None:
            from pool_inputs import Blocked
            from retry_chrono import valid_record
            for key, obj in last.items():
                if key in self.input_fingerprints and obj.get('input_sha256') != self.input_fingerprints[key]:
                    raise Blocked('Routed result input mismatch; refuse stale resume')
            return {k for k, obj in last.items() if k in self.input_fingerprints
                    and valid_record(obj)}
        return {k for k, obj in last.items() if not obj.get('skip_reason')}

    # ------------------------------------------------------------------
    # Main evaluation workflow
    # ------------------------------------------------------------------

    def run(self, qa_file, drama, qa_data=None):
        if qa_data is None:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)

        os.makedirs(self.output_dir, exist_ok=True)

        raw_log_path = os.path.join(self.output_dir, 'raw_responses.jsonl')
        results_path = os.path.join(self.output_dir, 'results.json')

        done_set = self._load_done_set(raw_log_path)
        print('[{}] 已完成 {} 题'.format(drama, len(done_set)))

        # Collect tasks to run (in order: B_order -> B_sort -> D_same -> multi_hop)
        tasks = []
        for qa_type in QA_TYPES:
            for qa_id, qa_item in qa_data.get(qa_type, {}).items():
                if (qa_type, str(qa_id)) not in done_set:
                    tasks.append((drama, qa_type, str(qa_id), qa_item))

        if self.limit > 0:
            tasks = tasks[:self.limit]
            print('[{}] --limit {}，只跑前 {} 题'.format(drama, self.limit, len(tasks)))

        print('[{}] 待跑 {} 题，并发 {}'.format(drama, len(tasks), self.num_workers))

        if tasks:
            with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
                futures = {
                    pool.submit(self._eval_one, d, qt, qid, qitem, raw_log_path): (qt, qid)
                    for d, qt, qid, qitem in tasks
                }
                for fut in as_completed(futures):
                    qt, qid = futures[fut]
                    try:
                        res = fut.result()
                        # Skip-type questions are not saved in raw in _eval_one; add
                        # here
                        if res.get('skip_reason'):
                            atomic_write_jsonl(raw_log_path, res, self._raw_log_lock)
                        tag = ('✓' if res.get('correct') else '✗') if res.get('pred') else '?'
                        print('[{}:{}] {} pred={} ans={} tokens={}{}'.format(
                            qt, qid, tag,
                            res.get('pred'), res.get('answer'),
                            res.get('total_tokens', 0),
                            ' (guessed)' if res.get('guessed') else '',
                        ))
                    except Exception as e:
                        print('[ERROR] {}:{}: {}'.format(qt, qid, e))
        else:
            print('[{}] 所有题目已完成'.format(drama))

        self._save_summary(qa_data, drama, raw_log_path, results_path)

    # ------------------------------------------------------------------
    # Summarize and print accuracy
    # ------------------------------------------------------------------

    def _save_summary(self, qa_data, drama, raw_log_path, results_path):
        # Reconstruct result dictionary from jsonl, using the last entry for the same
        # key, supports re-running and overwriting
        result_map = {}
        if os.path.isfile(raw_log_path):
            try:
                with open(raw_log_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            k = (obj.get('qa_type', ''), str(obj.get('qa_id', '')))
                            result_map[k] = obj
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print('[WARN] 汇总读取失败: {}'.format(e))

        stats = {}
        total = correct = skipped = guessed = guessed_hit = 0

        for qa_type in QA_TYPES:
            items = qa_data.get(qa_type, {})
            tp = tot = 0
            for qa_id, qa_item in items.items():
                k = (qa_type, str(qa_id))
                res = result_map.get(k)
                if res is None:
                    continue
                if res.get('skip_reason'):
                    skipped += 1
                    continue
                tot += 1
                total += 1
                gt = qa_item.get('Answer', '')
                pred = res.get('pred') or res.get('result')
                is_ok = (pred == gt) if pred else False
                if is_ok:
                    tp += 1
                    correct += 1
                if res.get('guessed'):
                    guessed += 1
                    if is_ok:
                        guessed_hit += 1
            stats[qa_type] = dict(
                total=tot, correct=tp,
                acc=round(tp / tot * 100, 2) if tot else 0.0,
            )

        summary = dict(
            drama=drama, model=self.model,
            total=total, correct=correct,
            acc=round(correct / total * 100, 2) if total else 0.0,
            skipped=skipped, guessed=guessed, guessed_hit=guessed_hit,
            by_type=stats,
        )
        atomic_dump_json(summary, results_path)

        print('\n========== {} 评测汇总 =========='.format(drama))
        print('模型: {}'.format(self.model))
        print('总题数: {}  正确: {}  准确率: {:.2f}%  跳过(未计入): {}'.format(
            total, correct, summary['acc'], skipped))
        if guessed:
            real_acc = (correct - guessed_hit) / total * 100 if total else 0.0
            print('随机兜底: {} 题，猜中 {}；剔除猜中后准确率: {:.2f}%'.format(
                guessed, guessed_hit, real_acc))
        for qt, s in stats.items():
            print('  {}: {}/{} = {:.2f}%'.format(
                qt, s['correct'], s['total'], s['acc']))
        print('结果已写入: {}'.format(results_path))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Multi-Point QA 评测（所有参数通过命令行传入，无环境变量依赖）'
    )
    p.add_argument('--base-url',    required=True,
                   help='OpenAI-compatible API base URL.')
    p.add_argument('--api-key',     default='',
                   help='API key; prefer OPENAI_API_KEY to keep credentials out of process arguments.')
    p.add_argument('--keys-file',   default='',
                   help='File containing one API key per line; mutually exclusive with --api-key.')
    p.add_argument('--model',       required=True,
                   help='Served model ID.')
    p.add_argument('--qa-file',     required=True,
                   help='Path to a drama QA JSON file.')
    p.add_argument('--image-root',  required=True,
                   help='Frame root: <root>/<drama>/<window>/render/*.jpg.')
    p.add_argument('--output-dir',  required=True,
                   help='Output directory.')
    p.add_argument('--drama',       default='',
                   help='Drama name; defaults to the QA filename stem.')
    p.add_argument('--num-workers', type=int, default=2,
                   help='Concurrent inference workers (default: 2).')
    p.add_argument('--tpm-limit',   type=int, default=80000,
                   help='Token-per-minute limit; 0 disables throttling.')
    p.add_argument('--max-tokens',  type=int, default=1000,
                   help='Maximum output tokens per request.')
    p.add_argument('--max-retries', type=int, default=6,
                   help='Maximum API retries per request.')
    p.add_argument('--api-timeout', type=float, default=120.0,
                   help='API timeout in seconds.')
    p.add_argument('--limit',       type=int, default=0,
                   help='Legacy clip limit; 0 means all. Use dramavqa --limit for a question limit.')
    p.add_argument('--reasoning-effort', default='',
                   choices=('', 'low', 'high', 'max'),
                   help='GLM-5.3-Flash native thinking budget; omit for other models')
    return p.parse_args()


def main():
    args = parse_args()
    drama = args.drama or Path(args.qa_file).stem

    print('=' * 48)
    print('Multi-Point QA 评测')
    print('  模型      : {}'.format(args.model))
    print('  API 地址  : {}'.format(args.base_url))
    print('  QA 文件   : {}'.format(args.qa_file))
    print('  图片根目录: {}'.format(args.image_root))
    print('  结果目录  : {}'.format(args.output_dir))
    print('  剧目      : {}'.format(drama))
    print('  并发      : {}  TPM限速: {}'.format(args.num_workers, args.tpm_limit))
    print('=' * 48)

    ev = MultiPointEvaluator(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        image_root=args.image_root,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        tpm_limit=args.tpm_limit,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        limit=args.limit,
        keys_file=args.keys_file,
        reasoning_effort=args.reasoning_effort or None,
    )
    ev.run(args.qa_file, drama)


if __name__ == '__main__':
    main()
