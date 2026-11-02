import io
import gc
import re
import os
import cv2
import sys
import math
import time
import json
import random
import hashlib
import openai
import base64
import datetime
import argparse
import threading
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from functools import partial
from datetime import timedelta

# decord official only releases x86_64 wheels; aarch64 machines cannot install it (pypi
# mirror does not have eva-decord). When missing, fall back to cv2-based shim that
# maintains consistent RGB, frame count, and FPS, ensuring the rendering pipeline runs
# on both architectures. True decord is unaffected.
try:
    from decord import VideoReader, cpu
except ImportError:
    if __package__:
        from . import decord_cv2_shim
    else:
        import decord_cv2_shim

    decord_cv2_shim.install(force=True)
    from decord import VideoReader, cpu

    print('[info] decord 不可用，已启用 cv2 兜底解码 (decord_cv2_shim)')

# decord defaults to using CPU threads for decoding. With 176 CPU cores on the machine
# and 10+ concurrent processes during image generation, the thread count expands to
# thousands, leading to a load saturation but no actual output (long clips with large
# seek ranges will stall for tens of minutes). In practice, limiting the clip to a
# single thread reduces processing time to 42 seconds, thus converging the thread count.
DECORD_NUM_THREADS = int(os.environ.get('DECORD_NUM_THREADS', '2'))
from collections import defaultdict
from bisect import bisect_right
from PIL import Image, ImageFont, ImageDraw
from openai import OpenAI, DefaultHttpxClient
from concurrent.futures import ThreadPoolExecutor, as_completed

# gemini-related dependencies are optional: missing them does not affect the vllm/openai
# inference path.
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None






BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")

EPISODES_PER_SEASON = 365


# Short name -> model identifier.
#
# Only used for "direct external API" scenarios: gpt / gemini / doubao need to replace
# short name with the actual model ID of the service provider. Local vLLM path **does
# not use this** — the model name used in requests is overridden by
# VLLM_SERVED_MODEL_NAME (i.e., vllm serve --served-model-name), and the weight path is
# specified by MODEL_PATH.
#
# Unknown model names are passed through to the configured model service.
MODEL_CKPT_MAP = {
    "gpt-5": "gpt-5",
    "gpt-4.1": "gpt-4.1",
    "gpt-4o": "gpt-4o",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "conan": "conan",
    "long_rl": "long_rl",
    "doubao": "doubao",
    "qwen3.5-27b": "Qwen3.5-27B"
}

replace_names ={'Robert Crawley Earl of Grantham':'Robert Crawley',
                'Cora Crawley Countess of Grantham':'Cora Crawley',
                'Violet Crawley  Dowager Countess of Grantham':'Violet Crawley',
                'Doctor at Moorfields Hospital':'Doctor at Moorfields Hospital',
                'Violet Crawley- Dowager Countess of Grantham':'Violet Crawley',
                'Cora Crawley- Countess of Grantham':'Cora Crawley',
                'Robert Crawley- Earl of Grantham':'Robert Crawley',
                'General Sir Herbert Strutt':'Herbert Strutt',
                'Violet Crawley Dowager Countess of Grantham':'Violet Crawley',
                'The Velvet Violin Club Guest':'The Velvet Violin Club Guest',
                'Hugh Shrimpie MacClare Marquess of Flintshire':'Hugh Shrimpie MacClare',
                'Susan MacClare Marchioness of Flintshire':'Susan MacClare',
                "Daisy Mason": "Daisy Robinson",
                "Baxter": "Phyllis Baxter",
                "Lady Sybil Branson": "Lady Sybil Crawley",
                "Gwen Harding": "Gwen Dawson",
                '其他':'Others'}


DRAMA_CAP_MAP = {
    'the_long_night': 'chen_mo_de_zhen_xiang_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20260216.json',
    'ode_to_joy_1': 'huan_le_song_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251028.json',
    'minning_town': 'shan_hai_qing_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251101.json',
    'qin_empire_2': 'da_qin_di_guo_zhi_zong_heng_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',
    'the_knockout': 'kuang_biao_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251030.json',
    'stand_by_me_1': 'yi_qi_tong_guo_chuang_1_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',
    'downton_abbey': 'downton_abbey_hierarchical_caption_Hiera_0202_3API_briefadaptlength_20260216.json',
    'a_lifelong_journey': 'ren_shi_jian_hierarchical_caption_Hiera_1023_3API_briefadaptlength_20251028.json',
    'battle_of_changsha': 'zhan_chang_sha_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251230.json',
    'friends': 'friends_hierarchical_caption_Hiera_0205_3API_briefadaptlength_20260205.json',
    'three_body': 'san_ti_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20260331.json',
    'empresses_in_the_palace': 'zhen_huan_zhuan_hierarchical_caption_Hiera_1027_3API_briefadaptlength_20251028.json',
    'lost': 'lost_hierarchical_caption_Hiera_0205_3API_briefadaptlength_20260210.json'
}

# One media directory per drama; episode numbers continue across seasons.
DRAMA_NAME_MAP = {
    'the_long_night': ['the_long_night'],
    'ode_to_joy_1': ['ode_to_joy_1'],
    'minning_town': ['minning_town'],
    'qin_empire_2': ['qin_empire_2'],
    'the_knockout': ['the_knockout'],
    'stand_by_me_1': ['stand_by_me_1'],
    'downton_abbey': ['downton_abbey'],
    'a_lifelong_journey': ['a_lifelong_journey'],
    'battle_of_changsha': ['battle_of_changsha'],
    'friends': ['friends'],
    'three_body': ['three_body'],
    'empresses_in_the_palace': ['empresses_in_the_palace'],
    'lost': ['lost'],
    'breaking_bad': ['breaking_bad'],
    'prison_break': ['prison_break'],
}


# Subtitle source description: in two modes, the visual content differs, so the model
# must be explicitly informed. For the burned-in mode, dialogue appears at the bottom of
# the frame and must be prompted to look there; otherwise, it may be overlooked. In the
# text mode, there are no subtitles on screen, and dialogue is provided as text
# immediately following each frame. Incorrect input could mislead the model (causing it
# to search for a non-existent subtitle on the screen).
SUBTITLE_DESC_BURN = "These are the frames of a video, with subtitles rendered onto the images."
SUBTITLE_DESC_TEXT = ("These are the frames of a video. The subtitle line for each frame is "
                      "provided as text right after that frame.")
SUBTITLE_DESC_NONE = "These are the frames of a video."

# SUBTITLE_TEXT_MODE=1 indicates that subtitles are not burned-in and are sent as text
# alongside the image (corresponding to --subtitle_text). This is dynamically set by
# LLMInference.__init__ to ensure consistency between the system prompt and visual
# content.
SUBTITLE_TEXT_MODE = False
# SUBTITLE_NONE_MODE=1 means subtitles are neither burned into frames nor sent as text
# (exp5_nosub).
SUBTITLE_NONE_MODE = False
# Ablation experiment mode: '' / exp1_burn / exp2_nosub / exp3_gallery / exp4_speaker /
# exp5_nosub / exp6_langonly, also written back in __init__. Note: Except for
# exp6_langonly, the system prompt is not rewritten based on ablation_mode; the original
# full prompt is reused, with only the first sentence switching between burn, text, and
# no subtitle modes. exp6_langonly is language-only: user contains only the question and
# options, and the system does not describe frame images, subtitles, or facial
# descriptions.
ABLATION_MODE = ''


def _subtitle_desc():
    if SUBTITLE_NONE_MODE:
        return SUBTITLE_DESC_NONE
    return SUBTITLE_DESC_TEXT if SUBTITLE_TEXT_MODE else SUBTITLE_DESC_BURN


# system prompt header (task description), experiment 1/3 shared.
def _message_system_head():
    return (_subtitle_desc() +
            "\nSelect the best answer to the following single-choice question based on the video.")


MESSAGE_SYSTEM_HEAD = _message_system_head()

# output format only: no specific question answers included. Shared across all three
# experiments.
MESSAGE_SYSTEM_FORMAT = """
Respond with only the letter of the correct option, enclosed in angle brackets, e.g. <letter>."""

# specific example (only experiment 1 uses). Example shows format only, no actual answer
# letters provided.
MESSAGE_SYSTEM_EXAMPLE = """

Example:
Question: In the bedroom scene, what is the main color of Hillary's shirt?
A. White
B. Black
C. Green
D. Yellow
E. Blue
F. Purple
Answer: <letter>"""

# use model with thinking: allows reasoning, but must end with fixed format for
# consistent regex extraction. Enable with THINK_MODE=1.
def _think_prompt():
    return (_subtitle_desc() +
            "\nPlease perform a detailed reasoning based on the provided video frames to answer "
            "the following single-choice question, selecting the best option, and provide your "
            "final response strictly in the format: 'Final Answer: <letter>'.")


THINK_PROMPT = _think_prompt()

THINK_MODE = os.environ.get("THINK_MODE", "0").strip().lower() in ("1", "true", "yes")

MESSAGE_SYSTEM = MESSAGE_SYSTEM_HEAD + MESSAGE_SYSTEM_FORMAT + MESSAGE_SYSTEM_EXAMPLE

# exp6_langonly: No frame images, subtitles, or facial descriptions are provided; the
# system prompt also does not include these inputs.
def _langonly_message_system_head():
    return "Select the best answer to the following single-choice question."


def _langonly_think_prompt():
    return ("Please perform a detailed reasoning to answer the following "
            "single-choice question, selecting the best option, and provide your "
            "final response strictly in the format: 'Final Answer: <letter>'.")

# Only the full prompt is supported; THINK_MODE selects its reasoning variant.
SYS_PROMPT_MODE = "full"


def resolve_system_content(system_content):
    """Resolve the original full prompt after runtime subtitle settings are applied."""
    if ABLATION_MODE == 'exp6_langonly':
        if THINK_MODE:
            return _langonly_think_prompt()
        return _langonly_message_system_head() + MESSAGE_SYSTEM_FORMAT + MESSAGE_SYSTEM_EXAMPLE

    if THINK_MODE:
        return _think_prompt()
    is_group = system_content == GROUP_MESSAGE_SYSTEM
    head = _group_message_system_head() if is_group else _message_system_head()
    example = GROUP_MESSAGE_SYSTEM_EXAMPLE if is_group else MESSAGE_SYSTEM_EXAMPLE
    return head + MESSAGE_SYSTEM_FORMAT + example


def _group_message_system_head():
    return (_subtitle_desc() + """
Every 4 consecutive frames are tiled into one 2x2 composite image, in playback order:
- Top row, left to right: frames 1 and 2;
- Bottom row, left to right: frames 3 and 4.
So the top-left cell is the earliest frame of the group and the bottom-right cell is the latest. The composite images themselves are also ordered by playback time.
Select the best answer to the following single-choice question based on the video.""")


GROUP_MESSAGE_SYSTEM_HEAD = _group_message_system_head()

GROUP_MESSAGE_SYSTEM_EXAMPLE = MESSAGE_SYSTEM_EXAMPLE

GROUP_MESSAGE_SYSTEM = GROUP_MESSAGE_SYSTEM_HEAD + MESSAGE_SYSTEM_FORMAT + GROUP_MESSAGE_SYSTEM_EXAMPLE

GROUP_MESSAGE_SYSTEM_FORMAT = MESSAGE_SYSTEM_FORMAT

prompt_template = """Question: {Question}
{Options}"""


class NoVideoAvailable(Exception):
    """This episode has no available video (not transmitted or directory name mismatch).
    Defined separately to distinguish "data missing can be skipped" from "true program
    error" for main."""


# Maximum output per request limit. Originally set to 1000, this causes 100% truncation
# for reasoning-heavy models (Qwen3.5/3.6-27B, which require hundreds to thousands of
# tokens before the main text begins) — the response is cut off before reaching
# "Answer:", leading to incomplete extraction of letters and being randomly marked as
# answered, permanently marked as incorrect. Real-world measurements show
# completion_tokens median values differ by two orders of magnitude across models
# (Instruct 2 / 8B-thinking 82 / 32B-thinking 890 / 3.5-27B 2046), so must be set
# model-specific, hence configured via environment variables. Be careful not to increase
# too much: the backend has ~120s gateway timeout, larger quotas result in longer
# generation times and higher risk of exceeding the timeout.
API_MAX_TOKENS = int(os.environ.get('API_MAX_TOKENS', '1000'))


class _TokenBudget:
    """Token bucket rate limiting based on TPM quota for request pacing. Background:
    External gateway (e.g., Silicon Flow) applies TPM rate limiting; a single 256-frame
    request consumes 22,000 tokens, while 80,000 TPM limits only allow ~3.6 requests per
    minute. Simply reducing thread count doesn't work effectively—when thread count is
    fixed, actual throughput depends on response latency, and high latency causes
    immediate quota violation (429 error). Therefore, the client actively limits request
    rate by pre-allocating estimated token amounts. If the bucket doesn't have enough
    tokens, the request waits until tokens become available. This ensures long-term
    throughput converges within the quota regardless of concurrency or latency. Reserve
    15% safety margin because the server counts tokens including output, which may
    exceed our estimated values."""

    def __init__(self, tpm):
        self.capacity = max(tpm * 0.85, 1.0)
        self.tokens = self.capacity
        self.rate = self.capacity / 60.0
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self, need):
        # If a single request exceeds the bucket capacity, it is treated as a full
        # bucket; otherwise, it will indefinitely wait.
        need = min(need, self.capacity)
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity,
                                  self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= need:
                    self.tokens -= need
                    return
                wait = (need - self.tokens) / self.rate
            time.sleep(min(wait, 30) + random.uniform(0, 0.5))


_TPM_LIMIT = int(os.environ.get('API_TPM_LIMIT', '0'))
_token_budget = _TokenBudget(_TPM_LIMIT) if _TPM_LIMIT > 0 else None


def _estimate_tokens(msg):
    """Estimate token count for a single request to be used in token bucket allocation.
    Image estimation: 392x392 frame ≈ 85 tokens per frame (measured: 256-frame requests
    total 22,035 tokens, including text ≈ 700, so (22,035 - 700)/256 ≈ 83). Text
    estimated at 4 tokens per unit."""
    imgs = 0
    chars = 0
    for m in msg:
        c = m.get('content')
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'image_url':
                    imgs += 1
                else:
                    chars += len(part.get('text', ''))
    # Output buffer reserve: must match the actual max_tokens value; otherwise, the rate
    # limiter will underestimate the request cost.
    return imgs * 85 + chars // 4 + API_MAX_TOKENS


def atomic_dump_json(obj, path, indent=4):
    """Atomic disk write: write temporary file → fsync → os.replace to overwrite target.
    Direct open(path, "w") truncates the original file to 0 bytes. If json.dump fails
    due to disk full (or process killed), the original content is permanently lost.
    Real-world example: q_a_dict.json was cleared, and raw_responses.jsonl entries with
    key→question mapping lost, rendering all responses invalid. Temporary file and
    target file must be in the same directory (os.replace requires same filesystem). If
    write fails, the target file remains unchanged."""
    tmp_path = f'{path}.tmp.{os.getpid()}.{threading.get_ident()}'
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            # Without fsync, a power outage or crash may cause the replace operation to
            # occur before data is written to disk, leaving behind empty files.
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def is_item_done(qa):
    """Return whether a question has a final result. An unextractable model response counts
    as completed and wrong. A missing video or image marked with skip_reason must be
    retried after data arrives. A guessed answer is incomplete and is retried by
    default; SKIP_RETRY_GUESSED=1 disables that retry."""
    if 'result' not in qa:
        return False
    if qa.get('skip_reason'):
        return False
    # The re-run guardian only retries questions with invalid model answers; it does not
    # check the correct answer, and incorrect responses do not trigger re-runs. The
    # default behavior remains unchanged, with the scanner process explicitly enabling
    # RETRY_FAILED=1.
    if os.environ.get('RETRY_FAILED') == '1':
        result = qa.get('result')
        return (not qa.get('guessed') and isinstance(result, str)
                and len(result) == 1 and 'A' <= result <= 'Z')
    # guessed default re-run; set SKIP_RETRY_GUESSED=1 to skip
    if qa.get('guessed') and os.environ.get('SKIP_RETRY_GUESSED') != '1':
        return False
    return True


# options lines like "A. xxx" / "B、xxx", take the first character of each line to get
# the valid option set.
_OPTION_LINE_RE = re.compile(r'^\s*\**\s*([A-Z])\s*[.、,，):：]', re.MULTILINE)


def parse_option_letters(options):
    """Parse valid option letters for random fallback in case of extraction failure.
    Options are a single string (e.g., "A. xxx\\nB. xxx"), with variable number of
    options (seen A-G). Random must fall within real existing options; otherwise,
    invalid answers may appear in data with no real options. If parsing fails, return
    empty list; caller decides whether to fallback."""
    if isinstance(options, dict):
        return sorted(str(k).strip().upper()[:1] for k in options
                      if str(k).strip()[:1].isalpha())
    if not isinstance(options, str):
        return []
    letters = []
    for m in _OPTION_LINE_RE.finditer(options):
        ch = m.group(1).upper()
        if ch not in letters:
            letters.append(ch)
    return letters


# Answer-extraction patterns are ordered by priority and shared with
# tools/reextract_answers.py. Match explicit final-answer markers before permissive
# fallbacks, so option text such as "A. white" is not mistaken for the model choice.
ANSWER_PATTERNS = [
    # special box markers (GLM series)
    (r'<\|begin_of_box\|>\s*([A-Za-z])\s*<\|end_of_box\|>', 0),
    # End-of-response anchor for Answer/answer line: strict format convention, using $
    # to avoid model repetition of "answer" text
    (r'(?:Final\s*Answer|Answer|答案)\s*[:：]\s*\**\s*\[?<?\s*([A-Za-z])\s*>?\]?\s*[.、,，)]?\s*\**\s*$',
     re.IGNORECASE),
    # Letters must be followed by a boundary; do not cut repeated
    # <letter>/option/unknown into l/o/u.
    (r'Final\s*Answer\s*[:：]\s*<?\s*([A-Za-z])(?![A-Za-z0-9_])\s*>?', re.IGNORECASE),
    # <letter>X</letter> / <answer>X</answer>: Gemini uses paired tags for answers;
    # single-character <X> rules are not recognized for multi-character tag names, so
    # they must be listed separately.
    (r'<\s*(?:letter|answer|option|choice)\s*>\s*\**\s*([A-Za-z])\s*\**\s*<\s*/',
     re.IGNORECASE),
    (r'<\s*([A-Za-z])\s*>', 0),
    (r'\[\s*([A-Za-z])\s*\]', 0),
    (r'(?:Answer|答案)\s*[:：]\s*<?\s*([A-Za-z])\s*>?[.、,，)\s]', re.IGNORECASE),
    (r'<answer>\s*\**\s*([A-Za-z])\s*[.、,，)]', re.IGNORECASE),
]

# Only recognized when the entire response is a single isolated letter (using
# fullmatch); otherwise, the first letter of the option list is incorrectly identified
# as the answer.
_ANSWER_FULLMATCH = r'\**\s*([A-Za-z])\s*[.、,，)]?\s*\**'
if __package__:
    from .answer_extraction import extract_final_choice, VERSION as ANSWER_EXTRACTION_VERSION
else:
    from answer_extraction import extract_final_choice, VERSION as ANSWER_EXTRACTION_VERSION


def extract_answer_letter(text, *, options=None, reasoning_only=False):
    'Accept only explicitly concluded final answers and validate the options according to the task.'
    return extract_final_choice(text, options, reasoning_only=reasoning_only)


class LLMInference:
    def _resolve_enable_thinking(self):
        """Resolve --enable_thinking (auto/on/off) to True, False, or None. None omits the
        field. GLM toggles thinking through chat_template_kwargs on shared weights;
        Qwen3-VL Instruct and Thinking use separate weights, so auto mode omits the
        switch for Thinking weights. Explicit on/off overrides the inferred setting for
        controlled experiments."""
        mode = getattr(self, 'enable_thinking_mode', 'auto')
        if mode == 'on':
            return True
        if mode == 'off':
            return False

        name = (self.short_model_name or '').lower()
        model_id = (self.model_name or '').lower()
        blob = name + ' ' + model_id

        # Qwen's thinking is independent of weights, not controlled by parameter
        # switches, and not sent to avoid interference
        if 'qwen' in blob and 'thinking' in blob:
            return None
        # Qwen3-VL Instruct (multimodal vision) has independent weights, does not think
        # inherently, and siliconflow endpoint does not accept enable_thinking parameter
        # — sending it results in 400
        # （current model does not support parameter enable_thinking）。
        # Can only be left un-sent, maintaining original behavior
        if 'qwen' in blob and 'vl' in blob:
            return None
        # GLM same-weight dual-mode, default disabled thinking to ensure concise output
        # for regex extraction
        if 'glm' in blob:
            return False
        # Other models (e.g., Qwen3.5/3.6-27B in plain text) follow historical behavior:
        # explicitly disable thinking, which has been tested to clear reasoning_content
        # and significantly reduce token usage.
        return False

    def __init__(self, args):
        self.port = args.port
        self.drama = args.drama
        self.level = args.level
        self.limit = getattr(args, 'limit', 0)
        self.shard_index = getattr(args, 'shard_index', 0)
        self.shard_total = max(1, getattr(args, 'shard_total', 1) or 1)
        self.sample_frames = args.sample_frames
        self.scale_factor = args.scale_factor
        self.target_width = getattr(args, 'target_width', 0) or 0
        self.subtitle_text = getattr(args, 'subtitle_text', False)
        self.render_subtitle = args.render_subtitle
        self.ablation_mode = (
            getattr(args, 'ablation_mode', '')
            or os.environ.get('ABLATION_MODE', '')
        ).strip().lower()
        if self.ablation_mode == 'exp3_gallery':
            # Role library experiment: subtitles always as text with image; video frames
            # extracted from video_data without face bounding boxes
            self.subtitle_text = True
            self.render_subtitle = False
        elif self.ablation_mode == 'exp4_speaker':
            # In exp2, only add speaker labels on same frame/subtitles, no full dialogue
            # or system rewrite
            self.subtitle_text = True
            self.render_subtitle = False
            if any(os.environ.get(k) == '1' for k in
                   ('ABLATE_NO_SUBTITLE', 'ABLATE_BURN_SUBTITLE')):
                raise ValueError('exp4_speaker 不允许关闭文本字幕或在线烧录字幕')
        elif self.ablation_mode == 'exp5_nosub':
            # No subtitle experiment: only provide face frame images, no text subtitles
            # burned or sent
            self.subtitle_text = False
            self.render_subtitle = False
        elif self.ablation_mode == 'exp6_langonly':
            # language-only: do not send frame images, subtitles, or face data, only
            # send questions and options
            self.subtitle_text = False
            self.render_subtitle = False
        global SUBTITLE_TEXT_MODE, SUBTITLE_NONE_MODE, ABLATION_MODE
        ABLATION_MODE = self.ablation_mode
        # subtitle description must match actual screen content: in nosub mode, no
        # subtitles are displayed, dialogue text is sent as text per frame; using the
        # burned-in version description would cause the model to attempt to find
        # non-existent subtitles on screen. SingleHopEval missed this backfill
        # (description always uses burned-in version), corrected here.
        SUBTITLE_NONE_MODE = self.ablation_mode in ('exp5_nosub', 'exp6_langonly')
        SUBTITLE_TEXT_MODE = bool(self.subtitle_text and not self.render_subtitle)
        image_root = Path(args.image_path)
        if image_root.name == self.drama and image_root.parent.name == 'img_data':
            if len(image_root.parts) < 5 or image_root.parts[-5] != self.level:
                raise ValueError('expanded image path must end in <level>/<frames>/<results>/img_data/<drama>')
            self.sample_frames = int(image_root.parts[-4])
        else:
            image_root = (image_root / self.level / str(self.sample_frames)
                          / self._results_dirname() / 'img_data' / self.drama)
        self.image_path = str(image_root)

        self.short_model_name = args.model_name
        # when using local vLLM service, the model name in the request must be the name
        # registered in the server via --served-model-name, not the model path
        # (otherwise 404: model does not exist). Priority: VLLM_SERVED_MODEL_NAME >
        # --model_path > MODEL_CKPT_MAP > short name.
        #
        # MODEL_CKPT_MAP is only meaningful for the "directly connect to external API"
        # scenario (gpt/gemini/doubao use it to obtain the real model ID). Values in
        # this table for local vLLM paths will be overridden by VLLM_SERVED_MODEL_NAME,
        # so use .get() instead of [] to avoid
        # KeyError during startup when running new models (e.g., glm-4.6v) that do not
        # need to be added to the table first.
        self.model_name = MODEL_CKPT_MAP.get(args.model_name, args.model_name)
        if args.model_path != "":
            self.model_name = args.model_path
        # Only local vLLM paths should use served-model-name to override; if misused on
        # API paths, it will incorrectly set the model ID to the local service name,
        # causing the provider to return "model not found".
        if (os.environ.get("VLLM_SERVED_MODEL_NAME")
                and (getattr(args, 'serving_mode', '')
                     or os.environ.get('SERVING_MODE', '')).strip().lower() != 'api'):
            self.model_name = os.environ["VLLM_SERVED_MODEL_NAME"]
        self.question_path = args.question_path
        assert self.drama in self.question_path
        assert '.json' in self.question_path
        self.video_paths = [os.path.join(args.video_path, drama_) for drama_ in DRAMA_NAME_MAP[self.drama]]
        self.speaker_root = os.path.abspath(
            getattr(args, 'speaker_root', '') or os.environ.get('SPEAKER_ROOT')
            or os.path.join(os.path.dirname(os.path.abspath(args.video_path)), 'speaker_data'))
        self._speaker_cache = {}
        self._speaker_sources = {}
        self._speaker_cache_lock = threading.Lock()
        if self.ablation_mode == 'exp4_speaker' and not os.path.isdir(self.speaker_root):
            raise FileNotFoundError('说话人数据根目录不存在: ' + self.speaker_root)

        # thinking switch: parses to the final value to be sent, None indicates not
        # sending this field. Must distinguish between "sending False" and "not
        # sending": - For GLM series, the same weights are toggled via enable_thinking;
        # explicit True/False must be sent. - For Qwen3-VL, Instruct and Thinking are
        # two separate weight sets; thinking behavior is determined by the weights
        # themselves, and sending an unrecognized chat_template_kwargs may be ignored.
        # Worse, setting False for *-Thinking weights may suppress thinking output.
        # Therefore, in auto mode, Qwen-Thinking is always not sent to maintain the
        # original behavior.
        self.enable_thinking_mode = getattr(args, 'enable_thinking', 'auto')
        self.enable_thinking = self._resolve_enable_thinking()

        self.use_render_info = args.use_render_info
        self.use_rendered_video = getattr(args, 'use_rendered_video', False)
        if self.ablation_mode == 'exp3_gallery':
            self.use_rendered_video = False
        self.parallel = args.parallel
        self.num_workers = args.num_workers
        resolution_dict = {
            'low': 'media_resolution_low',
            'medium': 'media_resolution_medium',
            'high': 'media_resolution_high',
            'ultra_high': 'media_resolution_ultra_high',
        }

        self.group_4x4 = args.group_4x4
        self.group_2x2 = args.group_2x2

        self.media_resolution = resolution_dict[args.media_resolution]
        # caption is no longer used (QA data already includes L2-L5 timestamps), only
        # retained when configuration exists.
        self.caption_path = ''
        if self.drama in DRAMA_CAP_MAP and getattr(args, 'caption_path', ''):
            self.caption_path = os.path.join(args.caption_path, DRAMA_CAP_MAP[self.drama])

        self.only_gen_img = args.only_gen_img
        self.result_path = args.result_path

        # Model original response disk output: When extraction fails, logs are cleared
        # by the platform, making it impossible to trace what the model actually
        # replied. Instead, raw responses are appended to raw_responses.jsonl, with a
        # lock used in multithreaded environments to ensure one line per record is
        # written without interleaving.
        self._raw_log_lock = threading.Lock()
        self._raw_log_path = None

        self.level_info = {
            'L2': {
                'frames': [],
                'times': [],
            },
            'L3': {
                'frames': [],
                'times': [],
            },
            'L4': {
                'frames': [],
                'times': [],
            },
            'L5': {
                'frames': [],
                'times': [],
            },
            'L6': {
                'frames': [],
                'times': [],
            },
        }

        # Service source (local=local deployment using vLLM / api=external API).
        #
        # No longer rely on substring matching of model name: qwen3-32b uses "qwen" in
        # its name and would be incorrectly identified as a local deployment, leading to
        # connection to localhost. Therefore, explicitly specify the serving mode via
        # --serving_mode or SERVING_MODE, falling back to old name inference only when
        # not specified.
        serving_mode = (getattr(args, 'serving_mode', '')
                        or os.environ.get('SERVING_MODE', '')).strip().lower()
        if serving_mode not in ('local', 'api'):
            looks_local = ('qwen' in self.short_model_name
                           or 'glm' in self.short_model_name
                           or 'intern' in self.short_model_name)
            serving_mode = 'local' if looks_local else 'api'
        self.serving_mode = serving_mode

        if self.serving_mode == 'local':
            # Local also must apply API_TIMEOUT; otherwise, even after 1200s of retry,
            # the SDK defaults to 600s. Network retries are handled by openai_inference,
            # and disabling SDK implicit retries for audit purposes.
            self.client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL") or "http://localhost:{}/v1".format(self.port),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                timeout=float(os.environ.get("LOCAL_API_TIMEOUT", "120")),
                max_retries=0)
        elif os.environ.get('API_BASE_URL'):
            # External API: Address and key are obtained from environment variables to
            # avoid hardcoding keys in code or command line (which would be logged into
            # platform logs). API_MODEL_ID is used to map short names to the actual
            # model ID of the service provider.
            api_key = os.environ.get('API_KEY') or API_KEY
            # Single request timeout: Some endpoints (e.g., GLM/Qwen *-Thinking) may
            # take a long time to respond for individual requests, causing the worker to
            # block indefinitely on a futex (tested to hang for 30 minutes without
            # response). Setting a timeout prevents such hung requests from causing the
            # entire shard to stall. Timeout-logged requests are caught by
            # APITimeoutError and proceed to retry logic; if retries fail, the task is
            # skipped and the worker continues.
            api_timeout = float(os.environ.get('API_TIMEOUT', '120'))
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.environ['API_BASE_URL'],
                http_client=DefaultHttpxClient(),
                timeout=api_timeout,
                max_retries=0,
            )
            env_model_id = os.environ.get('API_MODEL_ID')
            if env_model_id:
                # --model_name determines the output directory name, while API_MODEL_ID
                # specifies which actual model is used for the request. When these two
                # do not match, results are written to the directory named A but the
                # content comes from B, and all directory, short_name, and log entries
                # show A, only run_meta matches correctly—making it extremely hard to
                # detect. When the caller omits API_MODEL_ID during multi-model parallel
                # execution, this leads to all models running the default model in
                # .api_env (e.g., 6 models all run the default model). It's safer to
                # reject such runs outright.
                if (self.model_name and env_model_id != self.model_name
                        and os.environ.get('ALLOW_MODEL_ID_MISMATCH') != '1'):
                    raise SystemExit(
                        f"ERROR: --model_name ({self.model_name}) 与 "
                        f"API_MODEL_ID ({env_model_id}) 不一致。\n"
                        f"  结果会写进 {self.model_name} 的目录，实际却请求 "
                        f"{env_model_id}，属于静默污染。\n"
                        f"  确认无误可设 ALLOW_MODEL_ID_MISMATCH=1 跳过此检查。"
                    )
                self.model_name = env_model_id
        # elif 'gemini' in self.short_model_name:
        #     self.client = genai.Client(
        #         api_key=API_KEY,
        #         http_options={'api_version': 'v1alpha'})
        else:
            self.client = OpenAI(
                api_key=API_KEY,
                base_url=BASE_URL,
                http_client=DefaultHttpxClient()
            )

        self.episode_info = {}
        self.skipped_videos = []
        self.skipped_clips = []
        self._gallery_clip_cache = {}
        self._gallery_clip_lock = threading.Lock()
        self._role_map_cache = {}
        if self.only_gen_img or self.ablation_mode == 'exp3_gallery':
            lazy_meta = (self.ablation_mode == 'exp3_gallery' and not self.only_gen_img)
            cumulative_duration = 0.0
            for season, video_path in enumerate(self.video_paths):
                if self.use_rendered_video:
                    # New layout: <drama>/video_face_data/<active_version>/*.mp4. Old
                    # layout: <drama>/video_face_rendered/*.mp4 (flat structure).
                    video_dir = None
                    face_video_root = os.path.join(video_path, 'video_face_data')
                    if os.path.isdir(face_video_root):
                        video_dir = self._resolve_versioned_dir(
                            face_video_root, 'face_render_step', fallback_keyword='face_video')
                        if video_dir is None:
                            print(f"[WARN] video_face_data 下无可用版本目录，跳过: {face_video_root}")
                            self.skipped_videos.append({'path': face_video_root, 'reason': 'no_version_dir'})
                            continue
                    else:
                        legacy_dir = os.path.join(video_path, 'video_face_rendered')
                        if os.path.isdir(legacy_dir):
                            video_dir = legacy_dir
                    if video_dir is None:
                        # Skip the task if the episode is incomplete or the directory
                        # name does not match. Do not let a single missing episode
                        # prevent the entire evaluation round from starting.
                        print(f"[WARN] 人脸视频目录不存在，跳过: {face_video_root}")
                        self.skipped_videos.append({'path': face_video_root, 'reason': 'dir_not_found'})
                        continue
                    video_list = os.listdir(video_dir)
                    video_list.sort()
                else:
                    video_data = os.path.join(video_path, 'video_data')
                    if not os.path.exists(video_data):
                        print(f"[WARN] video_data 目录不存在，跳过: {video_data}")
                        self.skipped_videos.append({'path': video_data, 'reason': 'dir_not_found'})
                        continue
                    video_dir = self._resolve_versioned_dir(video_data, 'video_step')
                    if video_dir is None:
                        print(f"[WARN] video_data 下无子目录，跳过: {video_data}")
                        self.skipped_videos.append({'path': video_data, 'reason': 'no_subdir'})
                        continue
                    video_list = os.listdir(video_dir)
                    video_list.sort()

                for video_i in tqdm(video_list):
                    if '.mp4' in video_i or '.mkv' in video_i:
                        if 'fixed' in video_i:
                            continue
                        # The ID must be a pure number: video_data may contain leftover
                        # files like 0002_old.mkv, which would cause a ValueError when
                        # converting to int, crashing the entire show during
                        # initialization (tested with prison_break_4, resulting in the
                        # loss of the entire series).
                        _stem = os.path.splitext(video_i)[0]
                        if not _stem.isdigit():
                            print(f"[WARN] 集号非数字，跳过: {video_i}")
                            self.skipped_videos.append({
                                'path': os.path.join(video_dir, video_i),
                                'reason': 'non_numeric_episode',
                            })
                            continue
                        video_i_path = os.path.join(video_dir, video_i)
                        # The role library mode for inference only registers paths;
                        # fps/frame rate is opened per episode during frame extraction,
                        # avoiding scanning the entire video at startup.
                        if lazy_meta:
                            self.episode_info['{}_{}'.format(
                                season + 1, int(os.path.splitext(video_i)[0]))] = {
                                'fps': None,
                                'total_frames': None,
                                'start_time': cumulative_duration,
                                'video_path': video_i_path,
                            }
                            continue
                        # A single video failure (e.g., moov atom not found) skips only
                        # that episode; otherwise, a corrupted file causes the entire
                        # task to crash during initialization.
                        try:
                            vr = VideoReader(video_i_path, ctx=cpu(0), num_threads=DECORD_NUM_THREADS)
                            fps = vr.get_avg_fps()
                            total_frames = len(vr)
                        except Exception as e:
                            print(f"[WARN] 视频打不开，跳过: {video_i_path}  {type(e).__name__}: {str(e)[:120]}")
                            self.skipped_videos.append({
                                'path': video_i_path,
                                'reason': f'{type(e).__name__}: {str(e)[:200]}',
                            })
                            continue
                        duration = total_frames / fps
                        self.episode_info['{}_{}'.format(season+1, int(os.path.splitext(video_i)[0]))] = {
                            'fps': fps,
                            'total_frames': total_frames,
                            'start_time': cumulative_duration,
                            'video_path': video_i_path,
                        }
                        cumulative_duration += duration
                        del vr
                        gc.collect()

            if self.skipped_videos:
                print(f"[WARN] 共跳过 {len(self.skipped_videos)} 个视频/目录（缺失或损坏）")

            if len(self.episode_info) == 0:
                # If no usable video is available for this episode (no transmission or
                # directory name mismatch), raise a specific exception, which main
                # catches and exits with code 0 to prevent the start.sh episode loop
                # from halting entirely.
                raise NoVideoAvailable(
                    f"剧目 {self.drama} 无可用视频，已跳过。检查过的路径: "
                    f"{[v['path'] for v in self.skipped_videos][:5]}")

    def sample_list_equally(self, lst, n):
        if n <= 0:
            return []
        if n >= len(lst):
            return lst[:]
        indices = np.linspace(0, len(lst) - 1, num=n, dtype=int)
        return [lst[i] for i in indices]

    def genai_inference(self, msg, top_p=0.01, temperature=0.1):
        retries = 5
        version = self.model_name
        parts = []
        msg_system = msg[0]['content']
        question = msg[1]['content'][0]['text']
        if msg_system:
            parts.append(types.Part(text=msg_system))
        parts.append(types.Part(text=question))
        for i in range(1, len(msg[1]['content'])):
            msg_type = msg[1]['content'][i]['type']
            msg_data = msg[1]['content'][i][msg_type]
            if msg_type == 'text':
                parts.append(types.Part(text=msg_data))
            else:
                parts.append(types.Part(
                                inline_data=types.Blob(
                                    mime_type="image/jpeg",
                                    data=msg_data['url'].split('base64,')[1],
                                    ),
                                media_resolution={"level": self.media_resolution}
                            )
                )

        for _ in range(retries):
            try:
                print(version,top_p,temperature)
                return self.client.models.generate_content(
                            model=version,
                            contents=[
                                    types.Content(
                                        parts=parts,
                                    )
                                ],
                        )

            except openai.RateLimitError:
                time.sleep(15)
                # embed()
            except openai.OpenAIError as e:
                print(f'ERROR: {e}')
                if "403" in str(e) and _ < 2:
                    print(f"Received 403 error. Retrying in ... seconds...")
                elif "403" in str(e) and _ == 2:
                    version='gpt-5'
                else:
                    # Not a 403 error or failure on the last attempt.
                    return None
            except Exception as e:
                print(f'ERROR: {e}')
                time.sleep(15)

    def openai_inference(self, msg, top_p=0.01, temperature=0.1):
        # A single 256-frame request is approximately 20MB / 22,000 tokens, easily
        # exceeding the external gateway's TPM quota. Previous retries=2 + fixed 2s
        # sleep was insufficient: rate limiting windows are typically measured in
        # minutes, and two rapid retries will both fail, resulting in 61% of questions
        # returning empty results.
        retries = int(os.environ.get('API_MAX_RETRIES', '6'))
        version = self.model_name
        # Filter out system messages with empty content.
        send_msg = [m for m in msg if not (m.get("role") == "system" and not m.get("content"))]
        for attempt in range(retries):
            try:
                current_sys_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(current_sys_time, self.drama, self.sample_frames, self.level, version, top_p, temperature)

                # Apply TPM quota pacing (only effective if API_TPM_LIMIT is set). Place
                # inside try block, immediately after actual sending, to ensure retries
                # re-request quota.
                if _token_budget is not None:
                    _token_budget.acquire(_estimate_tokens(send_msg))

                # When enable_thinking is None, do not send this field (e.g., Qwen
                # *-Thinking has independent weight), otherwise pass extra_body to the
                # server.
                create_kwargs = dict(
                    model=version,
                    temperature=temperature,
                    top_p=top_p,
                    messages=send_msg,
                    max_tokens=API_MAX_TOKENS,
                )
                if self.enable_thinking is not None:
                    # The enable_thinking switch is distributed on two different chains:
                    # local vLLM (OpenAI-compatible layer) only recognizes
                    # extra_body.chat_template_kwargs.enable_thinking, while the
                    # top-level enable_thinking is discarded as unknown (log "fields ...
                    # ignored: {'enable_thinking'}"), making GLM thinking effectively
                    # useless. GLM defaults to enabling thinking, so setting
                    # enable_thinking=False disables it. For external APIs
                    # (siliconflow), Qwen3.5/3.6-27B has been tested to recognize the
                    # top-level enable_thinking, and setting it to False clears
                    # reasoning_content and sets completion_tokens=3. Therefore, local
                    # use chat_template_kwargs, while external APIs use the top-level
                    # enable_thinking.
                    if self.serving_mode == 'local':
                        create_kwargs['extra_body'] = {
                            "chat_template_kwargs": {
                                "enable_thinking": self.enable_thinking
                            }
                        }
                    else:
                        create_kwargs['extra_body'] = {
                            "enable_thinking": self.enable_thinking
                        }
                return self.client.chat.completions.create(**create_kwargs)

            except openai.RateLimitError as e:
                # Exponential backoff + jitter: 15/30/60/120/240s (max 240s). Jitter
                # prevents multiple workers from waking up simultaneously and flooding
                # the quota at once.
                wait = min(15 * (2 ** attempt), 240) + random.uniform(0, 5)
                print(f'RATE_LIMIT (attempt {attempt + 1}/{retries}), 等待 {wait:.0f}s: {str(e)[:120]}')
                time.sleep(wait)
            except openai.OpenAIError as e:
                print(f'ERROR: {e}')
                # 5xx is gateway jitter, should be retried; 4xx (except 429) retry is
                # meaningless.
                status = getattr(e, 'status_code', None)
                retryable = status is None or status >= 500
                if not retryable or attempt == retries - 1:
                    return None
                wait = min(10 * (2 ** attempt), 120) + random.uniform(0, 5)
                print(f'  可重试错误，等待 {wait:.0f}s 后重试 ({attempt + 1}/{retries})')
                time.sleep(wait)

        print(f"Failed after multiple retries.")
        # Must return None: callers take .choices from response object; returning a
        # string causes AttributeError and crashes the entire worker (original
        # implementation returns string, a clear bug).
        return None

    def update_qa_dict(self, dict_path):
        if os.path.isfile(dict_path):
            is_completed = True
            with open(dict_path, 'r', encoding='utf-8') as f:
                try:
                    q_a_dict = json.load(f)
                except json.JSONDecodeError as e:
                    # When disk is full, write in truncate mode, leaving a partial
                    # 0-byte/semi-complete q_a_dict.json. Previously, this was silently
                    # discarded, causing the entire task to fail and the outer layer to
                    # record rc=0, leading to the task pool's cursor advancing and these
                    # cells being permanently skipped (tested to fail 365 times).
                    #
                    # Intentionally do not delete or rename the file: run_pool.sh's
                    # flock fails, and the same cell may be processed by two workers
                    # simultaneously. If the file contains an intermediate state from
                    # another worker, deletion would erase the already-completed
                    # results. Instead, leave the corrupted file for manual inspection.
                    # Returning an empty dictionary allows the run to restart from
                    # scratch, and the bad file will be properly overwritten on disk.
                    print(f'WARNING: {dict_path} 解析失败({e})，'
                          f'本格按全新任务重跑（原文件保留，未删除）')
                    return {}, False
                if not q_a_dict:
                    return {}, False
                qa_list = []
                q_id_new = int(list(q_a_dict.keys())[-1])
                for qa_k in q_a_dict.keys():
                    qa_list.append(q_a_dict[qa_k]['Q&A']['Question'])
                for level, q_info in self.questions.items():
                    for q_key, q_a_batch in q_info.items():
                        for q_id, q_a in q_a_batch['Q&A'].items():
                            if q_a['Question'] not in qa_list:
                                q_id_new += 1
                                q_a_dict[q_id_new] = {
                                    'time': q_a_batch['time'],
                                    'Q&A': q_a,
                                    'level': level,
                                    'clip_id': q_a_batch.get('clip_id', q_key),
                                    'q_id': q_id,
                                }
                for q_i, q_info in q_a_dict.items():
                    if not is_item_done(q_info['Q&A']):
                        is_completed = False
                        break
            return q_a_dict, is_completed
        else:
            return {}, False

    def build_dict(self):
        q_a_dict = {}
        q_id_new = -1
        for level, q_info in self.questions.items():
            for q_key, q_a_batch in q_info.items():
                for q_id, q_a in q_a_batch['Q&A'].items():
                    q_id_new += 1
                    q_a_dict[q_id_new] = {
                        'time': q_a_batch['time'],
                        'Q&A': q_a,
                        'level': level,
                        # q_key is now "category#clip_id", disk writes the original
                        # clip_id, maintaining consistent field semantics with existing
                        # results files.
                        'clip_id': q_a_batch.get('clip_id', q_key),
                        'q_id': q_id,
                    }
        return q_a_dict

    def _results_dirname(self):
        """Root directory of image files, at the first level. New rule (target_width mode):
        results.w<width>_<sub|nosub> - Width is unified, scale_factor is no longer
        valid; use actual width instead. - Subtitle burning affects image content, must
        be reflected in directory; otherwise sub/nosub modes will conflict. Old rule
        (scale_factor mode): results.<scale> remains unchanged, allowing previous images
        to be reused without being overwritten."""
        if self.target_width:
            suffix = 'sub' if self.render_subtitle else 'nosub'
            return 'results.w{}_{}'.format(self.target_width, suffix)
        return 'results.{}'.format(self.scale_factor)

    def _resolve_versioned_dir(self, parent_dir, step_key, fallback_keyword=None):
        """Parse data directory with version.json, return path to active version
        subdirectory. New data layout introduces version management:
        <drama>/<data_dir>/version.json {"<step>": {"active_version": "<dir>"}}
        <drama>/<data_dir>/<version_dir>/... actual files If version.json is missing or
        damaged, fall back to selecting subdirectories by keyword/dictionary order to
        ensure old structure remains readable."""
        if not os.path.isdir(parent_dir):
            return None

        version_file = os.path.join(parent_dir, 'version.json')
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r', encoding='utf-8') as fr:
                    version_data = json.load(fr)
                active = (version_data.get(step_key) or {}).get('active_version')
                if active:
                    active_path = os.path.join(parent_dir, active)
                    if os.path.isdir(active_path):
                        return active_path
                    print(f"[WARN] version.json 指定的版本目录不存在: {active_path}")
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] version.json 解析失败({parent_dir}): {type(e).__name__}: {e}")

        subdirs = sorted(d for d in os.listdir(parent_dir)
                         if os.path.isdir(os.path.join(parent_dir, d)))
        if not subdirs:
            return None
        if fallback_keyword:
            for d in subdirs:
                if fallback_keyword in d:
                    return os.path.join(parent_dir, d)
        return os.path.join(parent_dir, subdirs[-1])

    def _get_drama_root(self, sea_epi):
        """Derive the root directory from video path. Videos may come from video_data/ or
        video_face_data/ (new version, with version subdirectories), and after uniform
        stripping, the face_data / subtitle_data / speaker_data directories can be
        located."""
        video_path = self.episode_info[sea_epi]['video_path']
        for marker in ('/video_face_data/', '/video_face_rendered/', '/video_data/'):
            if marker in video_path:
                return video_path.split(marker)[0]
        return os.path.dirname(os.path.dirname(video_path))

    def _get_qa_path(self):
        """Compute the output directory for qa_data. image_path follows the format
        <root>/<level>/<frames>/results.<scale>/img_data/<drama>, take the last three
        segments (level/frames/results.<scale>) and append them to result_path, avoiding
        hardcoding parts[3:] across different root directory depths."""
        p = Path(os.path.dirname(os.path.dirname(self.image_path)))
        result_path = os.path.join(self.result_path, *p.parts[-3:])
        return os.path.join(result_path, 'qa_data', self.drama, self.short_model_name)

    def dump_run_meta(self, qa_path, system_content):
        """Write a run_meta.json file in the result directory, recording key configuration
        for this run. q_a_dict.json only stores answers, and cannot be used to
        identify the thinking policy or rendered subtitle settings; therefore,
        the original prompt text and its MD5 hash are saved
        together to facilitate verification and cross-comparison. Each run is covered
        and written, with a constant size, not growing with the number of questions."""
        sys_text = system_content or ''
        meta = {
            'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sys_prompt_mode': SYS_PROMPT_MODE,
            'run_tag': os.environ.get('RUN_TAG', ''),
            'retry_pass_id': os.environ.get('RETRY_PASS_ID', ''),
            'retry_failed': os.environ.get('RETRY_FAILED') == '1',
            'think_mode': THINK_MODE,
            # thinking switch: mode is a parameter (auto/on/off), sent is the actual
            # sent value, sent=None means the chat_template_kwargs were not included in
            # the request.
            'enable_thinking_mode': getattr(self, 'enable_thinking_mode', 'auto'),
            'enable_thinking_sent': getattr(self, 'enable_thinking', None),
            'system_prompt': sys_text,
            'system_prompt_md5': hashlib.md5(sys_text.encode('utf-8')).hexdigest(),
            'system_prompt_len': len(sys_text),
            'system_prompt_injected': bool(sys_text),
            'user_prompt_template': prompt_template,
            'model_name_requested': self.model_name,
            'model_short_name': self.short_model_name,
            'port': self.port,
            'top_p': 0.01,
            'temperature': 0.1,
            'max_tokens': API_MAX_TOKENS,
            'api_timeout': float(os.environ.get('API_TIMEOUT',
                                                '600' if self.serving_mode == 'local' else '120')),
            'answer_extraction_version': ANSWER_EXTRACTION_VERSION,
            'drama': self.drama,
            'level': self.level,
            'sample_frames': self.sample_frames,
            'scale_factor': self.scale_factor,
            'target_width': self.target_width,
            'limit': self.limit,
            'group_2x2': getattr(self, 'group_2x2', False),
            'group_4x4': getattr(self, 'group_4x4', False),
            'use_render_info': self.use_render_info,
            'render_subtitle': self.render_subtitle,
            'subtitle_text': self.subtitle_text,
            'use_rendered_video': getattr(self, 'use_rendered_video', False),
            'ablation_mode': getattr(self, 'ablation_mode', ''),
            'speaker_root': self.speaker_root if self.ablation_mode == 'exp4_speaker' else None,
            'speaker_input_report': (os.path.join(qa_path, 'speaker_input_report.json')
                                     if self.ablation_mode == 'exp4_speaker' else None),
            'speaker_sources': (dict(sorted(self._speaker_sources.items()))
                                if self.ablation_mode == 'exp4_speaker' else {}),
            'image_path': self.image_path,
            'question_path': self.question_path,
            'answer_regex': [pattern for pattern, _flags in ANSWER_PATTERNS]
                            + [r'\A' + _ANSWER_FULLMATCH + r'\Z'],
            'raw_response_log': os.path.join(qa_path, 'raw_responses.jsonl'),
        }
        try:
            atomic_dump_json(meta, os.path.join(qa_path, 'run_meta.json'))
        except Exception as e:
            # The metadata writing should not interrupt evaluation
            print(f'[warn] run_meta.json 写入失败: {e}')

        # The original response and run_meta.json are in the same directory for easy
        # localization by experiment/scene/level
        self._raw_log_path = os.path.join(qa_path, 'raw_responses.jsonl')

    def dump_raw_response(self, key, attempt, raw, extracted, total_tokens, finish_reason=None, reasoning_content=None):
        """Append the original model response to disk for later troubleshooting. Write
        failures only trigger warnings, not exceptions: logs are not evaluation results,
        so they should not interrupt the entire inference process. reasoning_content:
        the reasoning process for GLM or reasoning-type models (separated from content),
        normally content is already clean and reasoning is not extracted, but saving a
        copy aids audit and traceability. Whether it actually works and whether
        think-on/off is enabled is controlled by truncating the content to the first
        4000 characters."""
        if not self._raw_log_path:
            return
        _rc = reasoning_content if isinstance(reasoning_content, str) else ''
        record = {
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key': key,
            'drama': self.drama,
            'level': self.level,
            'sample_frames': self.sample_frames,
            'attempt': attempt,
            'retry_pass_id': os.environ.get('RETRY_PASS_ID', ''),
            'max_tokens': API_MAX_TOKENS,
            'api_timeout': float(os.environ.get('API_TIMEOUT',
                                                '600' if self.serving_mode == 'local' else '120')),
            'answer_extraction_version': ANSWER_EXTRACTION_VERSION,
            'extracted': extracted,
            'extract_ok': extracted is not None,
            'finish_reason': finish_reason,
            'total_tokens': total_tokens,
            'raw_len': len(raw) if raw is not None else 0,
            'raw': raw,
            'reasoning_len': len(_rc),
            'reasoning_content': _rc[:4000],
        }
        try:
            with self._raw_log_lock:
                with open(self._raw_log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f'[warn] raw_responses.jsonl 写入失败: {e}')

    def single_inference(self):
        qa_path = self._get_qa_path()
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)

        self.dump_run_meta(qa_path, resolve_system_content(MESSAGE_SYSTEM))

        dict_path = os.path.join(qa_path, 'q_a_dict.json')
        q_a_dict, is_completed = self.update_qa_dict(dict_path)

        if is_completed == True:
            return q_a_dict

        if q_a_dict == {}:
            q_a_dict = self.build_dict()

        for key, q_info in q_a_dict.items():
            if is_item_done(q_info['Q&A']):
                continue
            if self.ablation_mode == 'exp6_langonly':
                _, updates = self._process_item_langonly(key, q_info)
                if updates is not None:
                    q_a_dict[key]['Q&A'].update(updates)
                    atomic_dump_json(q_a_dict, dict_path)
                continue
            time = q_info['time']
            time = self.get_level_time(time)
            if time is None:
                continue
            time_ = time.replace(':', '-')
            time_ = time_.replace('#', '-')

            image_file_path = os.path.join(self.image_path, time_, 'render' if self.use_render_info else 'raw')

            # Consistent with parallel branches: missing image clips are skipped to
            # prevent evaluation interruption
            if not os.path.isdir(image_file_path):
                print(f"[WARN] 图片目录不存在，跳过该题 {self.drama}/{self.level}/{key}: {image_file_path}")
                q_a_dict[key]['Q&A']['result'] = None
                q_a_dict[key]['Q&A']['total_tokens'] = 0
                q_a_dict[key]['Q&A']['an4ques'] = q_info['Q&A']['Question']
                q_a_dict[key]['Q&A']['cot'] = None
                q_a_dict[key]['Q&A']['skip_reason'] = 'images_not_found'
                continue

            image_files = os.listdir(image_file_path)
            # image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]))
            # image_files = sorted(image_files, key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
            image_files = sorted([f for f in image_files if not f.startswith('.')], key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
            base64_frames = []
            frame_names = []
            for image_file in image_files:
                jpg_path = os.path.join(image_file_path, image_file)
                img = Image.open(jpg_path)
                base64_frames.append(img)
                frame_names.append(image_file)

            if not base64_frames:
                print(f"[WARN] 图片目录为空，跳过该题 {self.drama}/{self.level}/{key}: {image_file_path}")
                q_a_dict[key]['Q&A']['result'] = None
                q_a_dict[key]['Q&A']['total_tokens'] = 0
                q_a_dict[key]['Q&A']['an4ques'] = q_info['Q&A']['Question']
                q_a_dict[key]['Q&A']['cot'] = None
                q_a_dict[key]['Q&A']['skip_reason'] = 'images_empty'
                continue

            if self.group_4x4:
                base64_frames, frame_names = self.get_frame_names_4x4(base64_frames, frame_names)
            elif self.group_2x2:
                base64_frames, frame_names = self.get_frame_names_2x2(base64_frames, frame_names)

            sub_map = self.load_subtitle_map(image_file_path)

            msg = [
                    {"role": "system", "content": resolve_system_content(MESSAGE_SYSTEM)},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt_template.format(Question=q_info['Q&A']['Question'],
                                                                        Options=q_info['Q&A']['Options'])}
                    ]},
                ]

            if self.short_model_name in ['gpt-4.1', 'gpt-5']:
                #selected_indices = np.round(np.linspace(0, len(base64_frames) - 1, 48)).astype(int).tolist()
                selected_indices = range(len(base64_frames))
            else:
                selected_indices = range(len(base64_frames))

            for idx in selected_indices:
                frame = base64_frames[idx]
                # Subtitles are already burned into the image during generation, so no
                # further rendering is needed
                frame = self._align_pil_to_factor(frame)
                buffered = io.BytesIO()
                frame.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()
                frame_base64 = base64.b64encode(image_bytes).decode('utf-8')
                msg[1]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
                # When subtitles are not burned in, send the dialogue text immediately
                # after the image, maintaining alignment between text and image.
                if sub_map:
                    sub_text = self.build_subtitle_text(sub_map, frame_names[idx])
                    if sub_text:
                        msg[1]["content"].append(
                            {"type": "text", "text": f"字幕: {sub_text}"})
                # if self.drama in ['the_knockout', 'lost']:
                #     text_info = ""
                #     if isinstance(frame_names[idx], list):
                #         for sub_idx, sub_frame_name in enumerate(frame_names[idx]):
                #             speaker_key = sub_frame_name[:-4]
                #             if speaker_info[speaker_key] != {}:
                # Inactive speaker-text assembly example retained for reference.
                #     else:
                #         speaker_key = frame_names[idx][:-4]
                #         if speaker_info[speaker_key] != {}:
                #             text_info += speaker_info[speaker_key]["text"] + ". "

                #     if text_info != "":
                #         msg[1]["content"].append(
                # Inactive subtitle-message construction example retained for reference.

            result, total_tokens, cot, api_skip = self.get_llm_response(msg, key, options=q_info['Q&A']['Options'])
            if api_skip:
                # Consistent with parallel branch: if the API does not receive results,
                # skip_reason is set and the task is skipped, no random fallback is
                # performed.
                q_a_dict[key]['Q&A']['result'] = None
                q_a_dict[key]['Q&A']['total_tokens'] = total_tokens
                q_a_dict[key]['Q&A']['an4ques'] = q_info['Q&A']['Question']
                q_a_dict[key]['Q&A']['cot'] = cot
                q_a_dict[key]['Q&A']['skip_reason'] = api_skip
                q_a_dict[key]['Q&A']['guessed'] = False
                q_a_dict[key]['Q&A']['guess_reason'] = None
                q_a_dict[key]['Q&A']['guess_from'] = None
                atomic_dump_json(q_a_dict, dict_path)
                continue
            result, guess_info = self.apply_guess_fallback(result, q_info['Q&A'], key)

            q_a_dict[key]['Q&A']['result'] = result
            q_a_dict[key]['Q&A']['total_tokens'] = total_tokens
            q_a_dict[key]['Q&A']['an4ques'] = q_info['Q&A']['Question']
            q_a_dict[key]['Q&A']['cot'] = cot
            # Like parallel branches: clear the previous skipped image mark after
            # successful execution
            q_a_dict[key]['Q&A']['skip_reason'] = None
            q_a_dict[key]['Q&A']['guessed'] = False
            q_a_dict[key]['Q&A']['guess_reason'] = None
            q_a_dict[key]['Q&A']['guess_from'] = None
            q_a_dict[key]['Q&A'].update(guess_info)

            atomic_dump_json(q_a_dict, dict_path)

        return q_a_dict

    def get_frame_names_4x4(self, base64_frames, frame_names):
        # Used to store the filename for the 4x4 large image, concatenated with the save
        # path for the large image
        output_dir = "output_4x4_grids"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # ================= Result Container ================= Store the path to the
        # concatenated large image (list of one dimension) Store the original file names
        # (2D nested list: [[group1...], [group2...]])
        frame_names_4x4 = []
        frame_names_grouped = []

        # ================= Constants Definition ================= Number of images per
        # group, rows, columns
        GROUP_SIZE = 16
        GRID_ROWS = 4
        GRID_COLS = 4

        total_images = len(base64_frames)

        # ================== Loop Processing ================== Step size is 16, iterate
        # through all images
        for i in range(0, total_images, GROUP_SIZE):
            # 1. Use slicing to get data for the current group
            batch_imgs = base64_frames[i : i + GROUP_SIZE]
            batch_names = frame_names[i : i + GROUP_SIZE]

            # If data is empty, stop
            if not batch_imgs:
                break

            # 2. 【Core Requirement】Add the file name list of the current group to the
            # large list, making frame_names_grouped structure: [['1.jpg'...'16.jpg'],
            # ['17.jpg'...'32.jpg'], ...]
            frame_names_grouped.append(batch_names)

            # Prepare canvas for stitching, get size of a single small image (assuming
            # all images have the same size, use the first one as reference)
            img_w, img_h = batch_imgs[0].size
            # Create large image: width = small image width × 4, height = small image
            # height × 4
            grid_img = Image.new('RGB', (img_w * GRID_COLS, img_h * GRID_ROWS), (255, 255, 255))

            # 4. Iterate over the current group's 16 images for layout
            for idx, img in enumerate(batch_imgs):
                # Calculate row index: idx // 4, column index: idx % 4 for a 4x4 grid
                # (0, 0, 0, 0, 1, 1...) and (0, 1, 2, 3, 0, 1...) respectively
                row = idx // GRID_COLS
                col = idx % GRID_COLS

                x = col * img_w
                y = row * img_h

                grid_img.paste(img, (x, y))

            # 5. Save the large image after concatenation
            group_index = i // GROUP_SIZE
            save_name = f"grid_group_{group_index:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)

            # grid_img.save(save_path)

            # 6. Record the path of the large image
            frame_names_4x4.append(grid_img)
        return frame_names_4x4, frame_names_grouped

    def get_frame_names_2x2(self, base64_frames, frame_names):
        # Used to store the filename for the generated 2x2 large image, and the path for
        # saving the concatenated image
        output_dir = "output_2x2_grids"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # ================= Result Container ================= Store the path to the
        # concatenated large image (list of one dimension) Store the original file names
        # (2D nested list: [[group1...], [group2...]])
        frame_names_2x2 = []
        frame_names_grouped = []

        # ================= Constants Definition ================= Number of images per
        # group, rows, columns
        GROUP_SIZE = 4
        GRID_ROWS = 2
        GRID_COLS = 2

        total_images = len(base64_frames)

        # ================== Loop Processing ================== Step size is 16, iterate
        # through all images
        for i in range(0, total_images, GROUP_SIZE):
            # 1. Use slicing to get data for the current group
            batch_imgs = base64_frames[i : i + GROUP_SIZE]
            batch_names = frame_names[i : i + GROUP_SIZE]

            # If data is empty, stop
            if not batch_imgs:
                break

            # 2. 【Core Requirement】Add the file name list of the current group to the
            # large list, making frame_names_grouped structure: [['1.jpg'...'16.jpg'],
            # ['17.jpg'...'32.jpg'], ...]
            frame_names_grouped.append(batch_names)

            # Prepare canvas for stitching, get size of a single small image (assuming
            # all images have the same size, use the first one as reference)
            img_w, img_h = batch_imgs[0].size
            # Create large image: width = small image width × 4, height = small image
            # height × 4
            grid_img = Image.new('RGB', (img_w * GRID_COLS, img_h * GRID_ROWS), (255, 255, 255))

            # 4. Iterate over the current group's 16 images for layout
            for idx, img in enumerate(batch_imgs):
                # Calculate the row index by dividing idx by 2 (0, 0, 0, 0, 1, 1...) and
                # the column index by taking modulo 2 (0, 1, 2, 3, 0, 1...)
                row = idx // GRID_COLS
                col = idx % GRID_COLS

                x = col * img_w
                y = row * img_h

                grid_img.paste(img, (x, y))

            # 5. Save the large image after concatenation
            group_index = i // GROUP_SIZE
            save_name = f"grid_group_{group_index:03d}.jpg"
            save_path = os.path.join(output_dir, save_name)

            # grid_img.save(save_path)

            # 6. Record the path of the large image
            frame_names_2x2.append(grid_img)
        return frame_names_2x2, frame_names_grouped


    def parallel_inference(self):
        qa_path = self._get_qa_path()
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)

        self.dump_run_meta(qa_path, resolve_system_content(MESSAGE_SYSTEM))

        dict_path = os.path.join(qa_path, 'q_a_dict.json')
        q_a_dict, is_completed = self.update_qa_dict(dict_path)

        if is_completed == True:
            return q_a_dict

        # return q_a_dict

        if q_a_dict == {}:
            q_a_dict = self.build_dict()

        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:  # Adjust max_workers as needed
            for key, q_info in q_a_dict.items():
                futures.append(executor.submit(self.process_item, key, q_info))

            for future in as_completed(futures):
                try:
                    key, updates = future.result()
                    if updates is not None:
                        if os.environ.get('RETRY_PASS_ID'):
                            updates['retry_pass_id'] = os.environ['RETRY_PASS_ID']
                            updates['retry_updated_at'] = datetime.datetime.now().isoformat()
                            updates['retry_max_tokens'] = API_MAX_TOKENS
                            updates['retry_api_timeout'] = float(os.environ.get(
                                'API_TIMEOUT', '600' if self.serving_mode == 'local' else '120'))
                            updates['answer_extraction_version'] = ANSWER_EXTRACTION_VERSION
                        q_a_dict[key]['Q&A'].update(updates)
                        atomic_dump_json(q_a_dict, dict_path)
                except Exception as e:
                    print(f"[ERROR] process_item failed: {e}")
                    import traceback
                    traceback.print_exc()
        return q_a_dict

    # ------------------------------------------------------------------
    # exp3_gallery: role library + video_data without frame borders + subtitle text
    # ------------------------------------------------------------------

    def _pil_to_image_part(self, pil_img):
        pil_img = self._align_pil_to_factor(pil_img)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {"type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{}".format(frame_base64)}}

    def _load_subtitle_json(self, drama_root, episode):
        subtitle_path = os.path.join(drama_root, 'subtitle_data')
        try:
            with open(os.path.join(subtitle_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
            subtitle_file = version_data['subtitle_checking_step']['active_version']
            subtitle_file_path = os.path.join(subtitle_path, subtitle_file, episode + '.json')
            with open(subtitle_file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError) as e:
            print('[WARN] subtitle not found for {} episode {}: {}'.format(
                self.drama, episode, e))
            return None

    def _load_role_mapping(self, drama_root):
        if drama_root in self._role_map_cache:
            return self._role_map_cache[drama_root]
        mapping_name = {}
        speaker_path = os.path.join(drama_root, 'speaker_data')
        try:
            with open(os.path.join(speaker_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
            speaker_file = version_data['speaker_recognition_step']['active_version']
            speak_map_path = os.path.join(
                speaker_path, speaker_file, '.workspace/new_roles_map.json')
            with open(speak_map_path, 'r', encoding='utf-8') as f:
                mapping_name = json.load(f)
        except (FileNotFoundError, KeyError, json.JSONDecodeError, OSError, TypeError):
            mapping_name = {}
        self._role_map_cache[drama_root] = mapping_name
        return mapping_name

    def _normalize_role_name(self, role_name, episode, mapping_name):
        if not role_name:
            return ''
        if '{}'.format(episode) + '_pro' in role_name:
            return ''
        if '少年' in role_name:
            role_name = '少年' + role_name.replace('少年', '')
        elif '幼年' in role_name:
            role_name = '幼年' + role_name.replace('幼年', '')
        elif '童年' in role_name:
            role_name = '童年' + role_name.replace('童年', '')
        elif '成年' in role_name:
            role_name = '成年' + role_name.replace('成年', '')
        elif '青年' in role_name:
            role_name = '青年' + role_name.replace('青年', '')
        role_name = role_name.replace('（）', '').replace('()', '')
        if '新角色' in role_name or 'New Role' in role_name:
            role_name = mapping_name.get(role_name, '')
        if role_name in replace_names:
            role_name = replace_names[role_name]
        if role_name in ('其他', 'Others', ''):
            return ''
        return role_name

    def _crop_face_from_obj(self, video_path, obj_info, box_scale=1.5):
        geo = obj_info.get('geometry', {})
        afl = geo.get('active_frame_list', {})
        if not afl:
            return None
        first_entry = afl.get('0', list(afl.values())[0])
        box = first_entry.get('box', {})
        try:
            x0, y0 = int(box['0']), int(box['1'])
            x1, y1 = int(box['2']), int(box['3'])
        except (KeyError, ValueError, TypeError):
            return None
        frame_no = int(obj_info.get('start_frame', 0))
        vr = None
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=DECORD_NUM_THREADS)
            total = len(vr)
            idx = max(0, min(total - 1, frame_no))
            img = Image.fromarray(vr.get_batch([idx]).asnumpy()[0])
        except Exception as e:
            print('[WARN] 角色库裁脸读帧失败 {}: {}'.format(video_path, e))
            return None
        finally:
            if vr is not None:
                del vr
        W, H = img.size
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_w = (x1 - x0) / 2.0 * box_scale
        half_h = (y1 - y0) / 2.0 * box_scale
        nx0 = max(0, int(cx - half_w))
        ny0 = max(0, int(cy - half_h))
        nx1 = min(W, int(cx + half_w))
        ny1 = min(H, int(cy + half_h))
        if nx1 <= nx0 or ny1 <= ny0:
            return None
        return img.crop((nx0, ny0, nx1, ny1))

    def _sorted_episodes_in_l3(self, time_str):
        t_start_part, t_end_part = time_str.split('_')
        s_season, s_ep, s_time = self._parse_time_str(t_start_part)
        e_season, e_ep, e_time = self._parse_time_str(t_end_part)
        sorted_episodes = []
        for key in self.episode_info.keys():
            parts = key.split('_')
            if len(parts) == 2:
                sorted_episodes.append((key, int(parts[0]), int(parts[1])))
        sorted_episodes.sort(key=lambda x: (x[1], x[2]))
        start_idx = end_idx = -1
        for i, (key, season_n, ep_n) in enumerate(sorted_episodes):
            if season_n == s_season and ep_n == s_ep:
                start_idx = i
            if season_n == e_season and ep_n == e_ep:
                end_idx = i
        if start_idx < 0 or end_idx < 0 or start_idx > end_idx:
            return [], s_time, e_time
        return sorted_episodes[start_idx:end_idx + 1], s_time, e_time

    def _load_face_gallery_for_time(self, time_str):
        'In L3 time window, each role takes the first appearance of a face, and the bbox is enlarged by 1.5x.'
        episodes, s_time, e_time = self._sorted_episodes_in_l3(time_str)
        if not episodes:
            return []
        gallery = []
        seen = set()
        for i, (sea_epi, season_n, ep_n) in enumerate(episodes):
            video_path = self.episode_info[sea_epi]['video_path']
            c_start = s_time if i == 0 else None
            c_end = e_time if i == len(episodes) - 1 else None
            f_start, f_end = self.get_frame_range(sea_epi, video_path, c_start, c_end)
            if f_start is None:
                continue
            drama_root = self._get_drama_root(sea_epi)
            episode = str(ep_n).zfill(4)
            face_root = os.path.join(drama_root, 'face_data')
            face_dir = self._resolve_versioned_dir(
                face_root, 'face_recognition_step', fallback_keyword='insightface')
            if face_dir is None:
                continue
            face_path = os.path.join(face_dir, episode + '.json')
            try:
                with open(face_path, 'r', encoding='utf-8') as fr:
                    face_data = json.load(fr)
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                print('[WARN] face json 读失败 {}: {}'.format(face_path, e))
                continue
            mapping_name = self._load_role_mapping(drama_root)
            char_first = {}
            for _idx, obj_info in face_data.get('object_list', {}).items():
                frame_no = obj_info.get('start_frame', -1)
                if frame_no < f_start or frame_no > f_end:
                    continue
                id_list = obj_info.get('id_list') or {}
                if not id_list:
                    continue
                raw_name = id_list.get('0', {}).get('name', '')
                try:
                    score = float(id_list.get('0', {}).get('score', 1.0))
                except (TypeError, ValueError):
                    score = 1.0
                if score < 0.43:
                    continue
                name = self._normalize_role_name(raw_name, episode, mapping_name)
                if not name or name in seen:
                    continue
                if name not in char_first or frame_no < char_first[name]['start_frame']:
                    char_first[name] = obj_info
            for name, obj_info in sorted(
                    char_first.items(), key=lambda kv: kv[1].get('start_frame', 0)):
                if name in seen:
                    continue
                face_img = self._crop_face_from_obj(video_path, obj_info, box_scale=1.5)
                if face_img is None:
                    continue
                seen.add(name)
                gallery.append({'name': name, 'image': face_img})
        return gallery

    def _extract_gallery_clip(self, time_str):
        sample_res = self.sample_frames_from_video(
            time_str, save_path='', only_return_sample_res=True)
        frames = []
        frame_subs = []
        for sea_epi, images, frame_indices in sample_res:
            drama_root = self._get_drama_root(sea_epi)
            episode = sea_epi.split('_')[1].zfill(4)
            subtitle_data = self._load_subtitle_json(drama_root, episode)
            for im_id, frame_id in enumerate(frame_indices):
                arr = images[im_id]
                if self.target_width or self.scale_factor != 1:
                    arr = self.resize_img_pil(arr)
                frames.append(Image.fromarray(arr))
                sub = self.lookup_subtitle(subtitle_data, frame_id) if subtitle_data else ''
                frame_subs.append(sub or '')
        gallery = self._load_face_gallery_for_time(time_str)
        return {'gallery': gallery, 'frames': frames, 'frame_subs': frame_subs}

    def _get_gallery_clip(self, time_str):
        with self._gallery_clip_lock:
            cached = self._gallery_clip_cache.get(time_str)
            if cached is not None:
                return cached
        if __package__:
            from .gallery_cache import cache_path, identity, load
        else:
            from gallery_cache import cache_path, identity, load
        root = os.environ.get('GALLERY_CACHE_ROOT', str(Path(__file__).parent / 'cache/gallery'))
        path = cache_path(root, self, time_str)
        try:
            data = load(path, identity(self, time_str))
        except Exception as exc:
            raise RuntimeError(
                f'Gallery cache missing, stale or corrupt: {path}. '
                'Run dramavqa prepare --task ablation --mode exp3_gallery with matching inputs first.') from exc
        with self._gallery_clip_lock:
            self._gallery_clip_cache[time_str] = data
            return self._gallery_clip_cache[time_str]

    def _process_item_gallery(self, key, q_info):
        time_str = self.get_level_time(q_info['time'])
        if time_str is None:
            return key, None
        try:
            clip = self._get_gallery_clip(time_str)
        except Exception as e:
            print('[WARN] exp3_gallery 抽帧失败 {}/{} ({}): {}'.format(
                self.drama, key, time_str, e))
            return key, {'result': None, 'total_tokens': 0,
                         'an4ques': q_info['Q&A']['Question'],
                         'cot': None, 'skip_reason': 'video_extract_failed'}
        frames = clip.get('frames') or []
        if not frames:
            return key, {'result': None, 'total_tokens': 0,
                         'an4ques': q_info['Q&A']['Question'],
                         'cot': None, 'skip_reason': 'video_frames_empty'}

        msg = [
            {"role": "system", "content": resolve_system_content(MESSAGE_SYSTEM)},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_template.format(
                    Question=q_info['Q&A']['Question'],
                    Options=q_info['Q&A']['Options'])},
            ]},
        ]
        content = msg[1]["content"]
        gallery = clip.get('gallery') or []
        if gallery:
            content.append({"type": "text", "text": "=== Character Reference ==="})
            for item in gallery:
                content.append({"type": "text",
                                "text": "Character: {}".format(item['name'])})
                content.append(self._pil_to_image_part(item['image']))
        content.append({"type": "text", "text": "=== Video Frames ==="})
        frame_subs = clip.get('frame_subs') or []
        for idx, frame in enumerate(frames):
            content.append(self._pil_to_image_part(frame))
            sub_text = frame_subs[idx] if idx < len(frame_subs) else ''
            if sub_text:
                content.append({"type": "text", "text": "字幕: {}".format(sub_text)})

        result, total_tokens, cot, api_skip = self.get_llm_response(msg, key, options=q_info['Q&A']['Options'])
        if api_skip:
            return key, {'result': None, 'total_tokens': total_tokens,
                         'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                         'skip_reason': api_skip,
                         'guessed': False, 'guess_reason': None, 'guess_from': None}
        result, guess_info = self.apply_guess_fallback(result, q_info['Q&A'], key)
        updates = {'result': result, 'total_tokens': total_tokens,
                   'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                   'skip_reason': None,
                   'guessed': False, 'guess_reason': None, 'guess_from': None}
        updates.update(guess_info)
        return key, updates

    def _process_item_langonly(self, key, q_info):
        'Timing includes native request retries and answer extraction, does not include deployment or queue waiting.'
        if os.environ.get('LANGONLY_TIMING') != '1':
            return self._process_item_langonly_impl(key, q_info)
        started = time.perf_counter()
        updates = None
        error = None
        try:
            result_key, updates = self._process_item_langonly_impl(key, q_info)
            return result_key, updates
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            elapsed = time.perf_counter() - started
            if updates is not None:
                updates['timing_elapsed_s'] = elapsed
            answer = updates or {}
            status = ('failed' if error or answer.get('skip_reason') else
                      'guessed' if answer.get('guessed') else
                      'valid' if answer.get('result') else 'failed')
            record = dict(key=str(key), drama=self.drama, level=self.level,
                          model=self.short_model_name, elapsed_s=elapsed,
                          status=status, error=error or answer.get('skip_reason'))
            # Shared thread lock with native response logs; each question writes to disk
            # immediately, with exceptions preserved.
            with self._raw_log_lock:
                timing_path = os.path.join(self._get_qa_path(), 'langonly_timing.jsonl')
                with open(timing_path, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f'[langonly-time] {self.drama}/{key} {status} {elapsed:.3f}s', flush=True)

    def _process_item_langonly_impl(self, key, q_info):
        'exp6_langonly: user content contains only the question and options, without frame images/subtitles/face annotations.'
        msg = [
            {"role": "system", "content": resolve_system_content(MESSAGE_SYSTEM)},
            {"role": "user", "content": prompt_template.format(
                Question=q_info['Q&A']['Question'],
                Options=q_info['Q&A']['Options'])},
        ]
        result, total_tokens, cot, api_skip = self.get_llm_response(msg, key, options=q_info['Q&A']['Options'])
        if api_skip:
            return key, {'result': None, 'total_tokens': total_tokens,
                         'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                         'skip_reason': api_skip,
                         'guessed': False, 'guess_reason': None, 'guess_from': None}
        result, guess_info = self.apply_guess_fallback(result, q_info['Q&A'], key)
        updates = {'result': result, 'total_tokens': total_tokens,
                   'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                   'skip_reason': None,
                   'guessed': False, 'guess_reason': None, 'guess_from': None}
        updates.update(guess_info)
        return key, updates

    def process_item(self, key, q_info):
        if is_item_done(q_info['Q&A']):
            return key, None  # Return immediately with final result

        if self.ablation_mode == 'exp6_langonly':
            return self._process_item_langonly(key, q_info)

        if self.ablation_mode == 'exp3_gallery':
            return self._process_item_gallery(key, q_info)

        time = q_info['time']
        time = self.get_level_time(time)
        if time is None:
            return key, None
        time_ = time.replace(':', '-')
        time_ = time_.replace('#', '-')

        image_file_path = os.path.join(self.image_path, time_, 'render' if self.use_render_info else 'raw')
        # Skip video missing/damaged clips as no images are generated; mark reason to
        # avoid silent loss due to FileNotFoundError when using listdir
        if not os.path.isdir(image_file_path):
            print(f"[WARN] 图片目录不存在，跳过该题 {self.drama}/{self.level}/{key}: {image_file_path}")
            return key, {'result': None, 'total_tokens': 0,
                         'an4ques': q_info['Q&A']['Question'],
                         'cot': None, 'skip_reason': 'images_not_found'}
        image_files = os.listdir(image_file_path)
        # image_files = sorted(image_files, key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
        image_files = sorted([f for f in image_files if not f.startswith('.')], key=lambda x: tuple(map(int, x.split('.')[0].split('_'))))
        base64_frames = []
        frame_names = []

        for image_file in image_files:
            jpg_path = os.path.join(image_file_path, image_file)
            img = Image.open(jpg_path)
            base64_frames.append(img)
            frame_names.append(image_file)

        if not base64_frames:
            print(f"[WARN] 图片目录为空，跳过该题 {self.drama}/{self.level}/{key}: {image_file_path}")
            return key, {'result': None, 'total_tokens': 0,
                         'an4ques': q_info['Q&A']['Question'],
                         'cot': None, 'skip_reason': 'images_empty'}

        if self.group_4x4:
            base64_frames, frame_names = self.get_frame_names_4x4(base64_frames, frame_names)
        elif self.group_2x2:
                base64_frames, frame_names = self.get_frame_names_2x2(base64_frames, frame_names)

        sub_map = self.load_subtitle_map(image_file_path)

        msg = [
            {"role": "system", "content": resolve_system_content(MESSAGE_SYSTEM)},
            {"role": "user", "content": [
                {"type": "text", "text": prompt_template.format(Question=q_info['Q&A']['Question'],
                                                                Options=q_info['Q&A']['Options'])}]},
        ]

        if self.short_model_name in ['gpt-4.1', 'gpt-5']:
            # selected_indices = np.round(np.linspace(0, len(base64_frames) - 1, 48)).astype(int).tolist()
            selected_indices = range(len(base64_frames))
        else:
            selected_indices = range(len(base64_frames))

        for idx in selected_indices:
            frame = base64_frames[idx]

            # ABLATE_BURN_SUBTITLE=1: Burn subtitles.json text to screen during online
            # processing for alignment with old code's "burn subtitle" mode in ablation
            # experiments
            if os.environ.get('ABLATE_BURN_SUBTITLE') == '1':
                if not hasattr(self, '_ablate_sub_info'):
                    clip_dir = os.path.dirname(image_file_path.rstrip('/'))
                    self._ablate_sub_info = self.load_burn_subtitle_info(image_file_path) if True else {}
                    # Refresh each clip (using image_file_path as cache key)
                if not hasattr(self, '_ablate_sub_cache_key') or self._ablate_sub_cache_key != image_file_path:
                    self._ablate_sub_cache_key = image_file_path
                    self._ablate_sub_info = self.load_burn_subtitle_info(image_file_path)
                if self._ablate_sub_info:
                    frame = self._render_subtitle_on_pil(frame, frame_names[idx], self._ablate_sub_info)

            # Subtitles are already burned into the image during generation, so no
            # further rendering is needed
            frame = self._align_pil_to_factor(frame)
            buffered = io.BytesIO()
            frame.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()
            frame_base64 = base64.b64encode(image_bytes).decode('utf-8')
            msg[1]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}})
            # When subtitles are not burned in, send the dialogue text immediately after
            # the image, maintaining alignment between text and image.
            if sub_map:
                sub_text = self.build_subtitle_text(sub_map, frame_names[idx])
                if sub_text:
                    msg[1]["content"].append(
                        {"type": "text", "text": f"字幕: {sub_text}"})

            # if self.drama in ['the_knockout', 'lost']:
            #     text_info = ""
            #     if isinstance(frame_names[idx], list):
            #         for sub_idx, sub_frame_name in enumerate(frame_names[idx]):
            #             speaker_key = sub_frame_name[:-4]
            #             if speaker_info[speaker_key] != {}:
            # Inactive speaker-text assembly example retained for reference.
            #     else:
            #         speaker_key = frame_names[idx][:-4]
            #         if speaker_info[speaker_key] != {}:
            #             text_info += speaker_info[speaker_key]["text"] + ". "

            #     if text_info != "":
            #         msg[1]["content"].append(
            # Inactive subtitle-message construction example retained for reference.

        result, total_tokens, cot, api_skip = self.get_llm_response(msg, key, options=q_info['Q&A']['Options'])
        if api_skip:
            # If rate limiting or a 5xx error yields no API response, set skip_reason so
            # is_item_done treats the question as unevaluated and retries it later.
            # Never use random fallback here: that would fabricate a completed answer
            # and conceal the missing evaluation.
            return key, {'result': None, 'total_tokens': total_tokens,
                         'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                         'skip_reason': api_skip,
                         'guessed': False, 'guess_reason': None, 'guess_from': None}
        result, guess_info = self.apply_guess_fallback(result, q_info['Q&A'], key)
        # Explicitly clear skip_reason: after a previous skip due to missing images and
        # this round successfully completing, the old marker must be cleared to avoid
        # residual marks during update() merging, which could falsely indicate that
        # re-running is still needed.
        updates = {'result': result, 'total_tokens': total_tokens,
                   'an4ques': q_info['Q&A']['Question'], 'cot': cot,
                   'skip_reason': None,
                   # Explicitly reset: when successfully re-running and extracting an
                   # answer, the previous guess marker must be cleared.
                   'guessed': False, 'guess_reason': None, 'guess_from': None}
        updates.update(guess_info)
        return key, updates

    def apply_guess_fallback(self, result, qa, key):
        """When retries exhaust, pick a random valid option from the available choices. The
        purpose is to ensure each question has a result, preventing evaluation issues
        due to formatting problems. However, "guessed" and "correctly answered" must be
        distinguishable, so additional state is stored: guessed=True / guess_reason /
        guess_from (candidate letter set) Original responses are preserved in cot and
        raw_responses.jsonl, later can be reextracted offline using
        tools/reextract_answers.py with new regex patterns to overwrite."""
        if result is not None:
            return result, {}
        letters = parse_option_letters(qa.get('Options'))
        if not letters:
            # Unable to parse any options, random is meaningless; keep None and let
            # downstream treat it as incorrect answer.
            return None, {'guessed': False, 'guess_reason': 'no_options_parsed'}
        pick = random.choice(letters)
        print(f"[{key or 'Unknown'}] 提取失败，随机兜底选择 {pick} (候选 {''.join(letters)})")
        return pick, {'guessed': True, 'guess_reason': 'extract_failed',
                      'guess_from': ''.join(letters)}

    def get_llm_response(self, msg, key, max_retries=2, delay_base=0.5, options=None):
        last_raw = None
        for attempt in range(1, max_retries + 1):
            # openai_inference has already handled rate limiting and retry backoff
            # internally; returning None means no result was obtained. We cannot use the
            # original `while response is None` loop to retry, as that would lead to a
            # dead loop when quota is exhausted, and the original implementation relied
            # on returning a string to "break" the loop, causing callers to get
            # AttributeError when accessing .choices.
            response = self.openai_inference(msg)
            if response is None:
                print(f"[{key or 'Unknown'}] API 无响应（限流或服务端错误），记为未答")
                return None, 0, last_raw, 'api_no_response'

            result = response.choices[0].message.content
            reasoning_only = not bool((result or '').strip())
            # Usage is optional in compatible API responses; preserve unknown as null.
            total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
            # Some models (e.g., Qwen3.5/3.6-27B, reasoning-type) write entire body into
            # reasoning_content, with content always an empty string. These responses
            # cannot be treated as "no answer": - content is '' rather than None, so it
            # doesn't reach the is None branch; - empty string cannot extract letters,
            # so apply_guess_fallback will randomly generate a fake answer and mark it
            # as answered, permanently hiding the real result. Therefore, before
            # extraction, apply fallback by treating thinking content as the response
            # body. These models' enable_thinking/thinking switches on the server side
            # are ineffective, so they cannot output to content; only client-side
            # fallback is possible.
            if not (result or '').strip():
                reasoning = getattr(response.choices[0].message,
                                    'reasoning_content', None)
                if (reasoning or '').strip():
                    result = reasoning
            if result is None:
                print(f"[{key or 'Unknown'}] 响应 content 为空，记为未答")
                return None, total_tokens, last_raw, 'api_empty_content'
            stripped_result = result.strip()
            if not stripped_result:
                # content and reasoning_content are both empty. Previously, only checked
                # for is None, empty strings would propagate through and be marked as
                # answered, falsely reporting "model never output" as a wrong answer.
                # Using skip_reason allows is_item_done to be evaluated as not assessed,
                # so it can be re-run next time.
                print(f"[{key or 'Unknown'}] 响应内容为空串，记为未答")
                return None, total_tokens, last_raw, 'api_empty_content'
            last_raw = stripped_result
            # finish_reason distinguishes between "model finishes its own work" and "cut
            # off by max_tokens", the latter being a common cause of extraction failure.
            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            # For GLM and reasoning-type models: thinking is separated to
            # reasoning_content (content is clean answer). Save a copy for auditability
            # of whether thinking is effective; extraction still uses content only,
            # unaffected.
            _reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
            extracted = extract_answer_letter(
                stripped_result, options=options, reasoning_only=reasoning_only)
            if reasoning_only and finish_reason != 'stop':
                extracted = None
            match = extracted is not None
            self.dump_raw_response(key, attempt, stripped_result, extracted,
                                   total_tokens, finish_reason,
                                   reasoning_content=_reasoning)
            if match:
                return extracted, total_tokens, stripped_result, None  # Extraction successful, immediately return.

            # Extraction failed, preparing to retry (last attempt without waiting)
            if attempt < max_retries:
                wait_time = delay_base * (2 ** (attempt - 1))  # Exponential backoff
                print(f"[{key or 'Unknown'}] 第 {attempt} 次提取失败，响应: {repr(result)}，{wait_time:.2f}s 后重试...")
                time.sleep(wait_time)
            else:
                print(f"[{key or 'Unknown'}] 已达最大重试次数 ({max_retries})，仍无法提取有效格式。最终响应: {repr(result)}")

        # Extraction failed: result is set to None (downstream uses this to determine
        # incorrect response), but the last original reply is stored in cot for
        # persistence, avoiding reliance solely on logs for tracing. Note: skip_reason
        # is None: the model actually responded, but the format didn't match, which is
        # considered "evaluated" and should be counted as incorrect rather than retrying
        # (strictly differentiate from API response issues).
        return None, 0, last_raw, None

    def read_caption_info(self):
        """Deprecated: New QA data already includes L2-L5 timestamps directly, no longer
        need to read caption. Maintaining empty implementation for backward
        compatibility with old callers."""
        return

    def read_question_info(self):
        with open(self.question_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        # New QA format: {"low-level": {"106": {"L2_time": "...", "Q&A": {...}}}}; old
        # format: {"L2": {"clip_id": {"time": "...", "Q&A": {...}}}}; convert to old
        # format for consistency, maintaining subsequent processing flow unchanged.
        self.questions = {}
        level_key = f'{self.level}_time'

        # clip_id is unique within each category, but may conflict across categories;
        # q_id is also unique within each category (in practice, there are 298
        # overlapping q_id across groups with distinct questions). Therefore, keys must
        # include category, otherwise later clips will overwrite earlier ones, causing
        # silent disappearance of questions (in practice: mini2000 loses 9 questions,
        # full set loses 6954 questions).
        for category in ['low-level', 'high-level', 'multi-point']:
            if category not in qa_data:
                continue

            for clip_id, clip_data in qa_data[category].items():
                if level_key not in clip_data:
                    print(f"Warning: {level_key} not found in clip {clip_id}, skipping")
                    continue

                time = clip_data[level_key]

                if time not in self.level_info[self.level]['times']:
                    self.level_info[self.level]['times'].append(time)

                if self.level not in self.questions:
                    self.questions[self.level] = {}

                unique_key = f'{category}#{clip_id}'
                self.questions[self.level][unique_key] = {
                    'time': time,
                    'Q&A': clip_data['Q&A'],
                    'clip_id': clip_id,
                }

        # --limit: keep only the first N clips for quick smoke testing
        if self.limit and self.level in self.questions:
            kept = dict(list(self.questions[self.level].items())[:self.limit])
            print(f'[limit] 只处理 {len(kept)}/{len(self.questions[self.level])} 个 clip')
            self.questions[self.level] = kept

        # --shard_index/--shard_total: split the clips of this episode into shards and
        # distribute them across multiple processes for parallel generation. Use sorted
        # + modulo: sorted ensures consistent order across shards (dict order depends on
        # QA file reading order, although stable, it's not reliable to rely on it),
        # modulo instead of chunking ensures clips of varying lengths are evenly
        # distributed across shards, avoiding one shard becoming a long tail.
        if self.shard_total > 1 and self.level in self.questions:
            items = sorted(self.questions[self.level].items())
            kept = {k: v for i, (k, v) in enumerate(items)
                    if i % self.shard_total == self.shard_index}
            print(f'[shard {self.shard_index}/{self.shard_total}] '
                  f'本片处理 {len(kept)}/{len(items)} 个 clip')
            self.questions[self.level] = kept

        # exp3 requires offline-prepared, provenance-checked raw frames + faces + text.
        if self.ablation_mode == 'exp3_gallery':
            if __package__:
                from .gallery_cache import cache_path, identity, load
            else:
                from gallery_cache import cache_path, identity, load
            root = os.environ.get('GALLERY_CACHE_ROOT', str(Path(__file__).parent / 'cache/gallery'))
            for time_str in {q['time'] for rows in self.questions.values() for q in rows.values()}:
                path = cache_path(root, self, time_str)
                try:
                    load(path, identity(self, time_str), images=False)
                except Exception as exc:
                    raise RuntimeError(f'Gallery cache unavailable: {path}; run dramavqa prepare --task ablation --mode exp3_gallery first.') from exc
            return
        if self.ablation_mode == 'exp6_langonly':
            return
        if self.ablation_mode == 'exp4_speaker' and not self.only_gen_img:
            self.preflight_speaker_inputs()
            return

        for q_level, q_info in self.questions.items():
            for q_id in q_info.keys():
                time = q_info[q_id]['time']
                time = self.get_level_time(time)
                if time is None:
                    continue
                time_ = time.replace(':', '-')
                time_ = time_.replace('#', '-')
                sampled_img_path = os.path.join(self.image_path, time_)
                render_path = os.path.join(sampled_img_path, 'render')

                # 1. Check if sampling has already been completed
                is_already_sampled = (
                    os.path.exists(render_path) and
                    self.count_jpg_files(render_path) == self.sample_frames
                )

                sample_res = None # Initialize variables

                if not is_already_sampled:
                    current_sys_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('{}, drama: {}, level: {}, time period: {}, sample frames from video.....'.format(current_sys_time, self.drama, self.level, time))
                    # Single clip frame extraction failed (video missing/damaged/time
                    # interval falls within missing set) – skip only, do not interrupt
                    # entire project generation
                    try:
                        sample_res = self.sample_frames_from_video(time, sampled_img_path)
                    except Exception as e:
                        print(f"[WARN] clip 抽帧失败，跳过 {self.drama}/{self.level}/{q_id} ({time}): "
                              f"{type(e).__name__}: {str(e)[:160]}")
                        self.skipped_clips.append({
                            'drama': self.drama,
                            'level': self.level,
                            'clip_id': q_info[q_id].get('clip_id', q_id),
                            'time': time,
                            'reason': f'{type(e).__name__}: {str(e)[:200]}',
                        })
                        continue

                if is_already_sampled:
                    continue

        # Aggregate skip cases: use missing frames to identify which clips lack video
        if self.skipped_videos or self.skipped_clips:
            print(f"[SUMMARY] {self.drama}/{self.level}: 跳过视频 {len(self.skipped_videos)} 个, "
                  f"跳过 clip {len(self.skipped_clips)} 个")
            try:
                os.makedirs(self.image_path, exist_ok=True)
                # When splitting and parallel processing, multiple processes write to
                # same path; add shard suffix to avoid overwriting
                shard_sfx = ('' if self.shard_total <= 1
                             else f'_shard{self.shard_index}of{self.shard_total}')
                report = os.path.join(
                    self.image_path,
                    f'_skipped_{self.drama}_{self.level}_{self.sample_frames}{shard_sfx}.json')
                atomic_dump_json({
                    'drama': self.drama,
                    'level': self.level,
                    'sample_frames': self.sample_frames,
                    'skipped_videos': self.skipped_videos,
                    'skipped_clips': self.skipped_clips,
                }, report, indent=2)
                print(f"[SUMMARY] 跳过明细已写入 {report}")
            except Exception as e:
                print(f"[warn] 跳过明细写入失败: {e}")

    def count_jpg_files(self, sampled_img_path):
        image_extensions = ('.jpg', '.jpeg', '.png')
        # Exclude .tmp intermediate files: Multiple tasks may share the same image
        # directory (which is determined only by target_width/render_subtitle and
        # unrelated to SYS_PROMPT_MODE), including others' temporary files could cause
        # incorrect "whether sampling is complete" judgment.
        return sum(
            1 for f in os.listdir(sampled_img_path)
            if f.lower().endswith(image_extensions) and not f.startswith('.tmp')
        )

    def _imwrite_atomic(self, path, frame):
        """Atomic image writing: First write to a temporary file in the same directory,
        then use os.replace to rename it. Multiple evaluation runs
        share the same image directory and may generate images simultaneously.
        cv2.imwrite directly writing to a target path is not atomic, potentially
        resulting in only half-written JPEGs. os.replace is atomic on the same
        filesystem and ensures the target path either doesn't exist or is a complete
        image. Temporary filenames include pid and thread ID to avoid conflicts between
        different processes or threads. (Image generation is multithreaded, and multiple
        experiments may run in parallel as independent processes.) Note: Temporary files
        must retain .jpg extension; cv2.imwrite selects encoder based on extension, and
        ending with .tmp will result in "could not find a writer". Temporary filenames
        start with a dot, and the listdir filter on the inference side already excludes
        files starting with '.'."""
        base = os.path.basename(path)
        tmp_path = os.path.join(os.path.dirname(path),
                                '.tmp{}_{}_{}'.format(os.getpid(),
                                                      threading.get_ident(), base))
        try:
            ok = cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not ok:
                raise IOError('cv2.imwrite 返回 False: {}'.format(tmp_path))
            os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            raise
    def epi_time_to_global(self, epi_time_str):
        "Convert '0004-23:59:59.999' or 'S1#0003-23:59:59.999' to global seconds (float)."
        if '#' not in epi_time_str:
            # Compatible with old format (no season information), default season=1
            season = 1
            ep_and_time = epi_time_str
        else:
            season_str, ep_and_time = epi_time_str.split('#', 1)
            if not season_str.startswith('S'):
                raise ValueError(f"Invalid season format in '{epi_time_str}'")
            season = int(season_str[1:])  # S1 -> 1

        ep_str, time_str = ep_and_time.split('-', 1)
        episode = int(ep_str)
        h, m, s = time_str.split(':')
        local_sec = int(h) * 3600 + int(m) * 60 + float(s)

        total_episodes_before = (season - 1) * EPISODES_PER_SEASON + (episode - 1)
        return total_episodes_before * 86400.0 + local_sec

    def parse_time_range(self, time_range_str):
        "Parse '0003-23:59:00.000_0004-00:02:30.000' -> (start_global_sec, end_global_sec)"
        parts = time_range_str.split('_', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid time range format: '{time_range_str}'")

        start_part, end_part = parts
        start_sec = self.epi_time_to_global(start_part)
        end_sec = self.epi_time_to_global(end_part)

        # if start_sec > end_sec:
        #     raise ValueError(f"Invalid time range: start > end in '{time_range_str}'")

        return start_sec, end_sec

    def find_containing_source(self, source_times, target_time):
        """Find the source_time that completely contains target_time in source_times. Args:
        source_times: List[str], each like '0001-..._0002-...' target_time: str, like
        '0004-..._0005-...' Returns: str: the matching source segment Raises:
        RuntimeError: if not found (shouldn't happen per problem statement)"""
        tgt_start, tgt_end = self.parse_time_range(target_time)

        for seg in source_times:
            src_start, src_end = self.parse_time_range(seg)
            # Check full containment: [src_start, src_end] ⊇ [tgt_start, tgt_end]
            if src_start <= tgt_start + 10 and tgt_end <= src_end:
                return seg

        # raise RuntimeError(
        #     "No source segment contains the target time range. "
        #     "This contradicts the problem assumption."
        # )

    def get_level_time(self, time):
        'The time in the new QA data is already the corresponding interval for the current level, so return directly.'
        return time

    def _episode_meta(self, sea_epi, video_path):
        """Get (fps, total_frames) for the episode, preferentially using cached metadata
        from __init__. episode_info is already opened once for each episode during
        initialization, so reopening here is redundant—when L6 spans full series, a clip
        needs to iterate through hundreds of episodes, leading to significant overhead
        that scales to minutes. Only fallback to opening the video when cache is
        missing."""
        info = self.episode_info.get(sea_epi)
        if info and info.get('fps') and info.get('total_frames'):
            return info['fps'], info['total_frames']

        if not os.path.exists(video_path):
            print(f"[WARN] 视频不存在，跳过该集: {video_path}")
            return None, None
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=DECORD_NUM_THREADS)
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            del vr
            gc.collect()
            return fps, total_frames
        except Exception as e:
            print(f"[WARN] 视频打不开，跳过该集: {video_path}  {type(e).__name__}: {str(e)[:120]}")
            return None, None

    def get_frame_range(self, sea_epi, video_path, start_time=None, end_time=None):
        """Compute the closed interval [start_frame, end_frame] required to cover this
        episode. Returns only the boundaries, not expanded into a frame-by-frame list:
        L6 (the entire series) can have millions of frames, expanding all frames would
        consume GB-level memory, leading to OOM when using multiple processes. Frames
        missing or damaged return (None, None), which is skipped by the caller."""
        fps, total_frames = self._episode_meta(sea_epi, video_path)
        if not fps or not total_frames:
            return None, None

        if start_time is not None:
            start_frame = max(0, math.floor(self.time_str_to_seconds(start_time) * fps))
        else:
            start_frame = 0

        if end_time is not None:
            end_frame = math.ceil(self.time_str_to_seconds(end_time) * fps) - 1
            end_frame = min(total_frames - 1, end_frame)
        else:
            end_frame = total_frames - 1

        # Timestamp falls outside the episode (due to data annotation errors or video
        # cropping), the interval will be reversed, treated as no valid frames, rather
        # than causing subsequent linspace to raise an exception.
        if end_frame < start_frame:
            return None, None
        return start_frame, end_frame

    def get_raw_imgs(self, video_path, episode_id_str, start_time=None, end_time=None):
        'Frame-by-frame version, only for legacy callers; please use get_frame_range instead.'
        sea_epi = '_'.join([episode_id_str.split('_')[0],
                            str(int(episode_id_str.split('_')[1]))])
        start_frame, end_frame = self.get_frame_range(
            sea_epi, video_path, start_time, end_time)
        if start_frame is None:
            return []
        return [f"{episode_id_str}_{frame_id}"
                for frame_id in range(start_frame, end_frame + 1)]

    def _parse_time_str(self, time_part):
        """Parse a single timepoint string. Format: 'S5#0004-00:43:23.726' or
        '0004-00:43:23.726' (default is None season) Returns: (season_str, episode_int,
        time_str)"""
        if '#' in time_part:
            season_str, rest = time_part.split('#')
            # Extract 'S5' from the string
            season = int(season_str.replace('S', '').replace('s', ''))
        else:
            rest = time_part
            season = 1 # Or set as default value based on requirements

        episode_str, timestamp = rest.split('-')
        return season, int(episode_str), timestamp

    def sample_frames_from_video(self, time, save_path, only_return_sample_res=False):
        # 1. Example of parsing start and end time strings in time format:
        # 'S5#0004-00:43:23.726_S6#0001-00:05:10.123'
        t_start_part, t_end_part = time.split('_')

        # Extract (season, episode, timestamp string)
        s_season, s_ep, s_time = self._parse_time_str(t_start_part)
        e_season, e_ep, e_time = self._parse_time_str(t_end_part)

        # 2. Convert episode_info keys to ordered list; dictionary is unordered, we need
        # a list to determine the next episode for S5_0004 is S5_0005 or S6_0001; list
        # elements example: ('5_4', 5, 4) -> (original key, season int, episode int)
        sorted_episodes = []
        for key in self.episode_info.keys():
            # Assume key is in '5_6' format
            parts = key.split('_')
            if len(parts) == 2:
                season_num = int(parts[0])
                ep_num = int(parts[1])
                sorted_episodes.append((key, season_num, ep_num))

        # Sort by season and episode
        sorted_episodes.sort(key=lambda x: (x[1], x[2]))

        # Locate the start and end indices
        start_idx = -1
        end_idx = -1

        # Construct target key for matching (note: input is S5#0004, dictionary key is
        # '5_4', remove leading zeros)
        target_start_key = f"{s_season}_{s_ep}"
        target_end_key = f"{e_season}_{e_ep}"

        for i, (key, season_n, ep_n) in enumerate(sorted_episodes):
            # Comparison logic: compare keys directly or numeric values (more robust)
            if season_n == s_season and ep_n == s_ep:
                start_idx = i
            if season_n == e_season and ep_n == e_ep:
                end_idx = i

        if start_idx == -1 or end_idx == -1:
            raise ValueError(f"Error: Could not find start ({target_start_key}) or end ({target_end_key}) episode in episode_info.")
        if start_idx > end_idx:
            raise ValueError("Error: Start episode is after end episode.")

        # 4. Collect frame ranges for each episode within the interval (only record
        # boundaries, not expand)
        #    segments: [(sea_epi, season_n, ep_n, start_frame, end_frame, count)]
        segments = []
        total_available = 0
        for i in range(start_idx, end_idx + 1):
            key, season_n, ep_n = sorted_episodes[i]
            video_path = self.episode_info[key]['video_path']

            c_start_time = s_time if i == start_idx else None
            c_end_time = e_time if i == end_idx else None

            seg_start, seg_end = self.get_frame_range(
                key, video_path, c_start_time, c_end_time)
            if seg_start is None:
                # This episode is missing/damaged: skip it and continue concatenating
                # subsequent episodes; L6 cannot discard the entire clip just because
                # one episode is faulty
                continue

            count = seg_end - seg_start + 1
            segments.append((key, season_n, ep_n, seg_start, seg_end, count))
            total_available += count

        if total_available == 0:
            raise ValueError(
                f"区间内没有任何可用帧: {time} (drama={self.drama}, level={self.level})")

        # 5. Uniformly sample points on the virtual timeline after concatenation, then
        # map back to (episode, frame number). Equivalent to old linspace over full
        # frame list, but memory is O(number of episodes) rather than O(total frames):
        # L6 with millions of frames per episode, the former is a few KB, the latter
        # requires several GB
        picks = np.linspace(0, total_available - 1, self.sample_frames)
        picks = np.round(picks).astype(int)
        picks = np.clip(picks, 0, total_available - 1)

        # Global offset at the beginning of a segment, to facilitate binary search to
        # locate a specific video in the global index
        offsets = np.cumsum([0] + [seg[5] for seg in segments])
        frames_by_episode = defaultdict(list)
        for global_idx in picks.tolist():
            seg_i = int(np.searchsorted(offsets, global_idx, side='right') - 1)
            seg_i = min(seg_i, len(segments) - 1)
            key, season_n, ep_n, seg_start, seg_end, _ = segments[seg_i]
            frame_idx = seg_start + (global_idx - int(offsets[seg_i]))
            frame_idx = min(frame_idx, seg_end)
            frames_by_episode[key].append(frame_idx)

        sample_res = []

        # Iterate over the dictionary after grouping, processing one video at a time
        for sea_epi, frame_indices in frames_by_episode.items():
            # Get the video path
            video_path = self.episode_info[sea_epi]['video_path']

            # VideoReader construction can fail for a missing or corrupt file; keep it
            # inside try so finally does not reference an undefined vr.
            vr = None
            try:
                vr = VideoReader(video_path, ctx=cpu(0), num_threads=DECORD_NUM_THREADS)
                # Get all target frames in the video batch, note: frame_indices should
                # be sorted, as some VideoReader read orders improve efficiency
                frame_indices.sort()
                images = vr.get_batch(frame_indices).asnumpy()

                # Store results in a list
                sample_res.append([sea_epi, images, frame_indices])

            except Exception as e:
                print(f"Error processing {sea_epi} ({video_path}): {type(e).__name__}: {e}")
                # Failures for a single episode are skipped, other frames continue
                # generation

            finally:
                # Key step: after processing the current video, explicitly delete
                # references and reclaim memory
                if vr is not None:
                    del vr
                gc.collect()

        # vrs = {}
        # for selected_frame_indice in selected_frame_indices:
        #     season = selected_frame_indice.split('_')[0]
        #     episode = int(selected_frame_indice.split('_')[1])
        #     sea_epi = '{}_{}'.format(season, episode)
        #     if sea_epi in vrs.keys():
        #         continue
        #     video_path = self.episode_info[sea_epi]['video_path']
        #     vrs[sea_epi] = VideoReader(video_path)
        #     gc.collect()

        # sample_res = []
        # for vr_key in vrs.keys():
        #     sea_epi_frame_indices = []
        #     for selected_frame_indice in selected_frame_indices:
        #         season = selected_frame_indice.split('_')[0]
        #         episode = int(selected_frame_indice.split('_')[1])
        #         sea_epi = '{}_{}'.format(season, episode)
        #         if sea_epi == vr_key:
        #             sea_epi_frame_indices.append(int(selected_frame_indice.split('_')[2]))
        #     images = vrs[vr_key].get_batch(sea_epi_frame_indices).asnumpy()
        #     sample_res.append([vr_key, images, sea_epi_frame_indices])

        # del vrs
        # gc.collect() # Optional, force garbage collection

        # for sample_item in sample_res:
        #     episode = sample_item[0]
        #     images = sample_item[1]
        #     frame_indices = sample_item[2]
        #     for im_id in range(images.shape[0]):
        #         img = Image.fromarray(images [im_id])
        #         if self.scale_factor != 1:
        #             img = self.resize_img(img)
        #         img.save(os.path.join(save_raw_path,
        #                               str(episode).zfill(4) + '_' + str(frame_indices[im_id])+".jpg"))

        if only_return_sample_res:
            return sample_res
        self.render(sample_res, save_path)
        return sample_res

    def render(self, sample_res, save_path):
        def process_single_frame(im_id, images, frame_indices, frame_face_data_epi_sea,
                             subtitle_data, save_render_path, season, episode, sea_epi,
                             skip_face_boxes=False):
            """Handle rendering work for a single video frame. Args: im_id (int): index of
            the current frame images (numpy.ndarray): array containing all frames
            frame_indices (list): list of frame IDs corresponding to each frame
            frame_face_data_epi_sea (dict): dictionary storing face data for each frame
            subtitle_data (dict): dictionary storing subtitle data save_render_path
            (str): path to save rendered image season (str): season information episode
            (str): episode information sea_epi (str): season-episode combination
            identifier Returns: None: function does not return a value; the processed
            image is saved to the specified path"""
            frame = images[im_id]
            # Use local variables instead of instance attributes: Since this function is
            # executed concurrently by 128 threads, writing instance attributes may
            # overwrite each other, leading to incorrect bbox scaling ratios.
            orig_frame_width = frame.shape[1]
            if self.target_width or self.scale_factor != 1:
                frame = self.resize_img_pil(frame)

            frame_id = frame_indices[im_id]

            # Draw Face Boxes (When using pre-rendered face video, the frame already has
            # boxes, so no need to redraw)
            if not skip_face_boxes and frame_id in frame_face_data_epi_sea.keys():
                face_data = frame_face_data_epi_sea[frame_id]
                # bbox refers to coordinates in the original resolution; convert to
                # scaled width by dividing by original width. In target_width mode,
                # scale_factor no longer represents the actual scaling ratio.
                box_scale = frame.shape[1] / float(orig_frame_width or frame.shape[1])
                frame = self.draw_box(face_data, frame, 0.43, sea_epi, box_scale)

            # Render Subtitles
            if self.render_subtitle == True:
                frame = self.render_subtitles(frame, subtitle_data, frame_id)

            # Save Image
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Determine filename based on path structure
            if 'S' in save_render_path.split('/')[-2]:
                filename = f"{season}_{episode}_{frame_id}.jpg"
            else:
                filename = f"{episode}_{frame_id}.jpg"

            full_save_path = os.path.join(save_render_path, filename)
            self._imwrite_atomic(full_save_path, frame)

        for sample_item in sample_res:
            sea_epi = sample_item[0]
            images = sample_item[1]
            frame_indices = sample_item[2]

            video_path = self.episode_info[sea_epi]['video_path']
            drama_root = self._get_drama_root(sea_epi)

            season = sea_epi.split('_')[0]
            episode = sea_epi.split('_')[1].zfill(4)

            frame_face_data_epi_sea = {}
            if not self.use_rendered_video:
                face_root = os.path.join(drama_root, 'face_data')
                face_dir = self._resolve_versioned_dir(
                    face_root, 'face_recognition_step', fallback_keyword='insightface')
                if face_dir is None:
                    print(f'WARNING: face_data not found for {self.drama} episode {episode}')
                else:
                    face_path = os.path.join(face_dir, episode + '.json')
                    try:
                        with open(face_path, 'r', encoding='utf-8') as fr:
                            face_data = json.load(fr)
                            for obj_idx, obj_info in face_data['object_list'].items():
                                assert obj_info['start_frame'] == obj_info['end_frame']
                                if obj_info['start_frame'] in frame_face_data_epi_sea.keys():
                                    frame_face_data_epi_sea[obj_info['start_frame']].append(obj_info)
                                else:
                                    frame_face_data_epi_sea[obj_info['start_frame']] = [obj_info]
                    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
                        print(f'WARNING: face_data unreadable for {self.drama} episode {episode}: {e}')

            subtitle_data = None
            # When flashing is disabled but subtitle text is required, still need to
            # read subtitle to export mapping file
            if self.render_subtitle or self.subtitle_text:
                subtitle_path = os.path.join(drama_root, 'subtitle_data')
                try:
                    with open(os.path.join(subtitle_path, 'version.json'), 'r', encoding='utf-8') as fr:
                        version_data = json.load(fr)
                        subtitle_file = version_data['subtitle_checking_step']['active_version']
                        subtitle_file_path = os.path.join(subtitle_path, subtitle_file, episode+'.json')
                        with open(subtitle_file_path, 'r', encoding='utf-8') as file:
                            subtitle_data = json.load(file)
                except (FileNotFoundError, KeyError) as e:
                    print(f'WARNING: subtitle not found for {self.drama} episode {episode}: {e}')
                    subtitle_data = None

            save_render_path = os.path.join(save_path, 'render')
            os.makedirs(save_render_path, exist_ok=True)
            # --- Multi-threading Implementation ---
            # We use max_workers to limit threads (e.g., os.cpu_count() or a fixed number like 8 or 16)
            # Adjust max_workers based on your disk I/O speed and CPU cores.
            with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
                # Create a partial function with fixed arguments that don't change per frame
                worker = partial(process_single_frame,
                                images=images,
                                frame_indices=frame_indices,
                                frame_face_data_epi_sea=frame_face_data_epi_sea,
                                subtitle_data=subtitle_data,
                                save_render_path=save_render_path,
                                season=season,
                                episode=episode,
                                sea_epi=sea_epi,
                                skip_face_boxes=self.use_rendered_video)

                # Submit tasks for all image indices
                # map will execute the worker function for every index in the range
                executor.map(worker, range(images.shape[0]))

            # nosub + subtitle mode: store "image filename -> subtitle" mapping to disk,
            # during inference, send dialogue text as text alongside image in sequence
            if self.subtitle_text and not self.render_subtitle:
                sub_map = {}
                for im_id in range(images.shape[0]):
                    frame_id = frame_indices[im_id]
                    if 'S' in save_render_path.split('/')[-2]:
                        fname = f"{season}_{episode}_{frame_id}.jpg"
                    else:
                        fname = f"{episode}_{frame_id}.jpg"
                    sub_map[fname] = self.lookup_subtitle(subtitle_data, frame_id)
                map_path = os.path.join(save_path, 'subtitles.json')
                old = {}
                if os.path.exists(map_path):
                    try:
                        with open(map_path, 'r', encoding='utf-8') as fr:
                            old = json.load(fr)
                    except (json.JSONDecodeError, OSError):
                        old = {}
                old.update(sub_map)
                atomic_dump_json(old, map_path, indent=2)

    def _speaker_episode(self, frame_name):
        'Image name uses episode_frame or season_episode_frame; frame refers to the original frame number.'
        parts = os.path.splitext(os.path.basename(frame_name))[0].split('_')
        if len(parts) == 2:
            season, episode, frame_id = 1, int(parts[0]), int(parts[1])
        elif len(parts) == 3:
            season, episode, frame_id = map(int, parts)
        else:
            raise ValueError('无法解析说话人对应帧名: ' + frame_name)
        dramas = DRAMA_NAME_MAP[self.drama]
        if not 1 <= season <= len(dramas):
            raise ValueError('说话人季号越界: ' + frame_name)
        return dramas[season - 1], f'{episode:04d}', frame_id

    def _load_speaker_episode(self, drama, episode):
        cache_key = (drama, episode)
        with self._speaker_cache_lock:
            if cache_key in self._speaker_cache:
                return self._speaker_cache[cache_key]
            parent = Path(self.speaker_root) / drama / 'speaker_data'
            with (parent / 'version.json').open(encoding='utf-8') as fr:
                versions = json.load(fr)
            active = next(((versions.get(step) or {}).get('active_version')
                           for step in ('speaker_processing_step', 'speaker_recognition_step')
                           if (versions.get(step) or {}).get('active_version')), None)
            if not active:
                raise ValueError(f'没有激活的说话人版本: {parent}')
            source = parent / active / f'{episode}.json'
            raw = source.read_bytes()
            data = json.loads(raw)
            objects = data['object_list']
            if not isinstance(objects, dict):
                raise ValueError(f'object_list 必须是字典: {source}')
            intervals = []
            for obj in objects.values():
                start, end = int(obj['start_frame']), int(obj['end_frame'])
                if end < start:
                    raise ValueError(f'说话人区间倒置: {source}: {start}-{end}')
                text, role = obj.get('text') or '', obj.get('role') or ''
                if not isinstance(text, str) or not isinstance(role, str):
                    raise ValueError(f'说话人 text/role 必须为字符串: {source}')
                intervals.append((start, end, text.strip(), role.strip()))
            intervals.sort(key=lambda item: item[0])
            starts, max_ends = [], []
            max_end = -1
            for start, end, _, _ in intervals:
                starts.append(start)
                max_end = max(max_end, end)
                max_ends.append(max_end)
            value = (intervals, starts, max_ends)
            self._speaker_cache[cache_key] = value
            self._speaker_sources[f'{drama}/{episode}'] = {
                'path': str(source), 'active_version': active,
                'sha256': hashlib.sha256(raw).hexdigest(),
                'segments': len(intervals),
            }
            return value

    def _speaker_at_frame(self, frame_name, subtitle):
        drama, episode, frame_id = self._speaker_episode(frame_name)
        intervals, starts, max_ends = self._load_speaker_episode(drama, episode)
        # Reverse-check all intervals covering this frame, supporting overlapping
        # speech; still match subtitle text exactly, do not add other transcripts or
        # full transcriptions to exp2 input, and do not infer names based on questions
        # or answers.
        idx = bisect_right(starts, frame_id) - 1
        matched = []
        while idx >= 0 and max_ends[idx] >= frame_id:
            start, end, text, role = intervals[idx]
            if start <= frame_id <= end and text == subtitle:
                matched.append(role)
            idx -= 1
        roles = []
        for role in reversed(matched):
            if role and role.lower() not in ('unknown', 'others', '其他', '未知') and role not in roles:
                roles.append(role)
        return roles, bool(matched)

    def _annotate_speaker_subtitles(self, sub_map):
        annotated = {}
        stats = {'subtitle_frames': 0, 'matched_frames': 0, 'speaker_frames': 0}
        for frame_name, text in sub_map.items():
            text = (text or '').strip()
            if not text:
                annotated[frame_name] = ''
                continue
            stats['subtitle_frames'] += 1
            roles, matched = self._speaker_at_frame(frame_name, text)
            stats['matched_frames'] += int(matched)
            stats['speaker_frames'] += int(bool(roles))
            annotated[frame_name] = f"{' / '.join(roles)}: {text}" if roles else text
        return annotated, stats

    def preflight_speaker_inputs(self):
        'Validate all clips before API calls; fail to proceed, preventing silent degradation to exp2.'
        report = {'drama': self.drama, 'speaker_root': self.speaker_root,
                  'clips': 0, 'questions': 0, 'frames': 0,
                  'subtitle_frames': 0, 'matched_frames': 0, 'speaker_frames': 0,
                  'matching_policy': 'exact subtitle text within inclusive original frame interval',
                  'unmatched_policy': 'keep original subtitle without a speaker label',
                  'minimum_match_rate': 0.99,
                  'errors': [], 'warnings': [], 'examples': []}
        seen = set()
        for clips in self.questions.values():
            for clip in clips.values():
                report['questions'] += len(clip['Q&A'])
                time_str = self.get_level_time(clip['time'])
                if time_str in seen:
                    continue
                seen.add(time_str)
                report['clips'] += 1
                try:
                    clip_dir = Path(self.image_path) / time_str.replace(':', '-').replace('#', '-')
                    image_dir = clip_dir / ('render' if self.use_render_info else 'raw')
                    names = sorted(p.name for p in image_dir.iterdir()
                                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
                    if len(names) != self.sample_frames:
                        raise ValueError(f'帧数 {len(names)} != {self.sample_frames}: {image_dir}')
                    with (clip_dir / 'subtitles.json').open(encoding='utf-8') as fr:
                        raw_map = json.load(fr)
                    missing = set(names) - raw_map.keys()
                    if missing:
                        raise ValueError(f'{len(missing)} 帧缺字幕映射: {clip_dir}')
                    sub_map = {name: raw_map[name] for name in names}
                    # Even if the clip has no spoken text, the corresponding speaker
                    # file must exist.
                    for drama, episode in {self._speaker_episode(name)[:2] for name in names}:
                        self._load_speaker_episode(drama, episode)
                    annotated, stats = self._annotate_speaker_subtitles(sub_map)
                    report['frames'] += len(names)
                    for key, count in stats.items():
                        report[key] += count
                    if stats['matched_frames'] != stats['subtitle_frames']:
                        mismatches = []
                        for name, text in sub_map.items():
                            text = (text or '').strip()
                            if text and not self._speaker_at_frame(name, text)[1]:
                                drama, episode, frame_id = self._speaker_episode(name)
                                intervals, _, _ = self._load_speaker_episode(drama, episode)
                                candidates = [{'text': line, 'role': role, 'start': start, 'end': end}
                                              for start, end, line, role in intervals
                                              if start <= frame_id <= end]
                                mismatches.append({'frame': name, 'subtitle': text,
                                                   'candidates': candidates})
                        report.setdefault('unmatched_subtitles', []).append(
                            {'clip': time_str, 'frames': mismatches})
                        report['warnings'].append(
                            f'{len(mismatches)} 帧未匹配说话人，保留原字幕不加名字: {clip_dir}')
                    if len(report['examples']) < 3:
                        for name in names:
                            if annotated[name] != (sub_map[name] or '').strip():
                                report['examples'].append({'frame': name, 'text': annotated[name]})
                                break
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    report['errors'].append({'clip': time_str, 'error': str(exc)})
        report['match_rate'] = report['matched_frames'] / max(report['subtitle_frames'], 1)
        if report['subtitle_frames'] and report['match_rate'] < report['minimum_match_rate']:
            report['errors'].append({'error': '说话人字幕匹配率低于 99%，请检查版本/时间坐标'})
        if not report['speaker_frames']:
            report['errors'].append({'error': '没有任何帧成功增加说话人，拒绝运行实验四'})
        for warning in report['warnings']:
            print('[speaker warning] ' + warning)
        report['sources'] = dict(sorted(self._speaker_sources.items()))
        qa_path = self._get_qa_path()
        os.makedirs(qa_path, exist_ok=True)
        atomic_dump_json(report, os.path.join(qa_path, 'speaker_input_report.json'), indent=2)
        print('[speaker preflight] {}: {} questions, {} clips, {} frames, '
              '{}/{} subtitle frames with speakers, {} errors'.format(
                  self.drama, report['questions'], report['clips'], report['frames'],
                  report['speaker_frames'], report['subtitle_frames'], len(report['errors'])))
        if report['errors']:
            raise ValueError('实验四输入检查失败: ' + str(report['errors'][:3]))
        return report

    def load_subtitle_map(self, image_file_path):
        """Read the mapping from image name to subtitle text (nosub + subtitle_text mode).
        Images are located in <clip>/render/, and the mapping file is in
        <clip>/subtitles.json. When ABLATE_NO_SUBTITLE=1, return an empty mapping for
        "only visuals, no subtitles" experiments (to align with old code path, which did
        not include subtitle injection)."""
        if os.environ.get('ABLATE_NO_SUBTITLE') == '1':
            return {}
        if not (self.subtitle_text and not self.render_subtitle):
            return {}
        map_path = os.path.join(os.path.dirname(image_file_path.rstrip('/')), 'subtitles.json')
        try:
            with open(map_path, 'r', encoding='utf-8') as fr:
                sub_map = json.load(fr)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            if self.ablation_mode == 'exp4_speaker':
                raise
            return {}
        if self.ablation_mode == 'exp4_speaker':
            sub_map = {name: text for name, text in sub_map.items()
                       if os.path.isfile(os.path.join(image_file_path, name))}
            sub_map, _ = self._annotate_speaker_subtitles(sub_map)
        return sub_map

    def load_burn_subtitle_info(self, image_file_path):
        """Convert subtitles.json to the structure expected by _render_subtitle_on_pil. Its
        image-name keys and text values are reshaped into
        speaker_info[image_stem]["text"]."""
        map_path = os.path.join(os.path.dirname(image_file_path.rstrip('/')), 'subtitles.json')
        try:
            with open(map_path, 'r', encoding='utf-8') as fr:
                raw = json.load(fr)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}
        info = {}
        for name, text in raw.items():
            text = (text or '').strip()
            info[os.path.splitext(name)[0]] = {'text': text} if text else {}
        return info

    def build_subtitle_text(self, sub_map, frame_name_entry):
        'Format the subtitle text corresponding to the current image; return empty string if no subtitle.'
        if not sub_map:
            return ''
        if isinstance(frame_name_entry, list):
            parts = []
            for sub_idx, name in enumerate(frame_name_entry):
                text = (sub_map.get(name) or '').strip()
                if text:
                    parts.append('第{}个小图: {}'.format(sub_idx + 1, text))
            return '\n'.join(parts)
        return (sub_map.get(frame_name_entry) or '').strip()

    def lookup_subtitle(self, subtitle_data, frame_id):
        'Retrieve subtitle text for a specific frame; return empty string if no subtitle exists.'
        if not subtitle_data:
            return ''
        for sub_key, info in subtitle_data['object_list'].items():
            start_frame = info['start_frame']
            end_frame = info['end_frame']
            if start_frame <= int(frame_id) <= end_frame:
                return info['text'] or ''
            elif int(frame_id) < start_frame:
                return ''
        return ''

    def render_subtitles(self, frame, subtitle_data, frame_id):
        if not subtitle_data:
            return frame
        for sub_key, info in subtitle_data['object_list'].items():
            start_frame = info['start_frame']
            end_frame = info['end_frame']
            if start_frame <= int(frame_id) <= end_frame:
                sub_content = info['text']
                if sub_content:
                    frame = self.render_subtitle_impl(frame, sub_content)
                return frame
            # object_list is sorted by frame number ascending; if subtitle start frame
            # exceeds current frame, terminate early
            elif int(frame_id) < start_frame:
                return frame
        return frame

    def _render_subtitle_on_pil(self, pil_img, frame_name_entry, speaker_info):
        'Render subtitle text on a PIL Image, supporting single frame and grouped frames (2x2/4x4).'
        pil_img = pil_img.copy()
        img_w, img_h = pil_img.size

        if isinstance(frame_name_entry, list):
            if self.group_4x4:
                grid_cols, grid_rows = 4, 4
            elif self.group_2x2:
                grid_cols, grid_rows = 2, 2
            else:
                grid_cols, grid_rows = int(len(frame_name_entry) ** 0.5), int(len(frame_name_entry) ** 0.5)
            sub_w = img_w // grid_cols
            sub_h = img_h // grid_rows
            for sub_idx, sub_frame_name in enumerate(frame_name_entry):
                speaker_key = sub_frame_name[:-4]
                if speaker_info.get(speaker_key, {}) != {}:
                    text = speaker_info[speaker_key]["text"]
                    row = sub_idx // grid_cols
                    col = sub_idx % grid_cols
                    region_x = col * sub_w
                    region_y = row * sub_h
                    self._draw_text_on_pil(pil_img, text, region_x, region_y, sub_w, sub_h)
        else:
            speaker_key = frame_name_entry[:-4]
            if speaker_info.get(speaker_key, {}) != {}:
                text = speaker_info[speaker_key]["text"]
                self._draw_text_on_pil(pil_img, text, 0, 0, img_w, img_h)

        return pil_img

    def _draw_text_on_pil(self, pil_img, text, region_x, region_y, region_w, region_h):
        'Render a subtitle centered at the bottom of a specified region on an PIL Image with a border.'
        draw = ImageDraw.Draw(pil_img)
        font_size = self._subtitle_font_size(region_w)
        try:
            font = ImageFont.truetype(os.environ.get('EVAL_FONT_PATH', 'DejaVuSans.ttf'), font_size)
        except IOError:
            font = ImageFont.load_default()

        lines = self._wrap_text_to_width(text, font, int(region_w * 0.96), draw)
        line_sizes = []
        total_height = 0
        line_gap = max(2, int(font_size * 0.25))
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            line_sizes.append((text_w, text_h))
            total_height += text_h + line_gap
        total_height -= line_gap

        margin_bottom = max(3, int(font_size * 0.35))
        start_y = region_y + region_h - total_height - margin_bottom
        start_y = max(start_y, region_y)

        stroke_width = max(1, int(font_size / 12))
        y_offset = start_y
        for i, line in enumerate(lines):
            text_w, text_h = line_sizes[i]
            x = region_x + (region_w - text_w) // 2
            for dx in (-stroke_width, 0, stroke_width):
                for dy in (-stroke_width, 0, stroke_width):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y_offset + dy), line, fill=(0, 0, 0), font=font)
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += text_h + line_gap

    # Subtitle font size: after uniforming width to target_width, the calculated font
    # size becomes too small (392px results in only 10px), old code using max(...,20) as
    # a fallback causes 20px to fill the screen. Here, directly setting font size to
    # 16px per line at 392px results in about 24 Chinese characters per line, and two
    # lines occupy only 21%~27% of screen height, balancing readability and overlap.
    SUBTITLE_FONT_SIZE_AT_392 = 16

    def _subtitle_font_size(self, img_w):
        if self.target_width:
            # Scale font size proportionally to 392 baseline, ensuring consistent sizing
            # across different widths
            size = int(round(self.SUBTITLE_FONT_SIZE_AT_392 * img_w / 392.0))
            return max(size, 12)
        return max(int(50 * (img_w / 1920.0)), 20)

    def _wrap_text_to_width(self, text, font, max_width, draw=None):
        """Break the subtitle into lines based on pixel width. The original implementation
        only breaks lines without width constraints, causing the subtitle to be
        truncated when it exceeds the width (approximately 24 characters for 392px
        width). Chinese characters do not have spaces for word segmentation, so they are
        measured character by character; English text is broken by spaces to avoid
        splitting words."""
        def measure(s):
            if draw is not None:
                bbox = draw.textbbox((0, 0), s, font=font)
            else:
                bbox = font.getbbox(s)
            return bbox[2] - bbox[0]

        lines = []
        for raw_line in (text.split('\n') if '\n' in text else [text]):
            if not raw_line:
                continue
            if measure(raw_line) <= max_width:
                lines.append(raw_line)
                continue
            cur = ''
            for ch in raw_line:
                trial = cur + ch
                if cur and measure(trial) > max_width:
                    # Prefer breaking lines at spaces to avoid splitting English words
                    if ' ' in cur.strip() and ch != ' ':
                        cut = cur.rstrip().rfind(' ')
                        if cut > 0:
                            lines.append(cur[:cut])
                            cur = cur[cut + 1:] + ch
                            continue
                    lines.append(cur)
                    cur = ch
                else:
                    cur = trial
            if cur:
                lines.append(cur)
        return lines or ['']

    def render_subtitle_impl(self, frame, sub_content):
        """Render a Chinese subtitle (multi-line) onto a frame. Args: frame (np.ndarray):
        OpenCV format image (BGR or RGB, but recommended to pass in RGB) sub_content
        (str): Subtitle content, can include newlines Returns: np.ndarray: The new image
        with the subtitle added (same channel order as input)"""
        # Assume frame is RGB (if not, ensure compatibility with PIL)
        h, w = frame.shape[:2]
        scale = w / 1920.0
        font_size = self._subtitle_font_size(w)
        try:
            font = ImageFont.truetype(os.environ.get('EVAL_FONT_PATH', 'DejaVuSans.ttf'), font_size)
        except IOError:
            # Fall back to default font if SimHei is not found (may not support Chinese)
            font = ImageFont.load_default()

        # Convert to PIL Image (note: PIL uses RGB format)
        if frame.shape[2] == 3:
            pil_img = Image.fromarray(frame)
        else:
            pil_img = Image.fromarray(frame, mode='L')  # Grayscale image processing (usually not used)

        draw = ImageDraw.Draw(pil_img)

        # Supports multi-line subtitles: automatically wraps to fit screen width, with
        # 4% margin to avoid edge contact
        max_text_width = int(w * 0.96)
        lines = self._wrap_text_to_width(sub_content, font, max_text_width, draw)

        # Calculate text size per line
        line_sizes = []
        max_width = 0
        total_height = 0
        line_gap = max(2, int(font_size * 0.25))
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            line_sizes.append((text_w, text_h))
            max_width = max(max_width, text_w)
            total_height += text_h + line_gap  # Line spacing

        total_height -= line_gap  # Last line does not require extra spacing

        # Subtitle position: centered at bottom (with some margin)
        margin_bottom = max(4, int(font_size * 0.5))
        start_y = h - total_height - margin_bottom
        start_y = max(start_y, 0)  # Prevent text from going beyond the top of the image

        # Draw each row with a black outline for better readability
        stroke_width = max(1, int(font_size / 12))
        y_offset = start_y
        for i, line in enumerate(lines):
            text_w, text_h = line_sizes[i]
            x = (w - text_w) // 2

            # Draw the black outline first (multiple offsets to simulate a thick border)
            for dx in (-stroke_width, 0, stroke_width):
                for dy in (-stroke_width, 0, stroke_width):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y_offset + dy), line, fill=(0, 0, 0), font=font)

            # Then draw the white text
            draw.text((x, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += text_h + line_gap

        # Convert back to numpy array
        return np.array(pil_img)

    def time_str_to_seconds(self, time_str):
        "Convert a time string in 'HH:MM:SS.mmm' format to seconds (float)"
        hms, ms = time_str.split('.')
        h, m, s = map(int, hms.split(':'))
        total_sec = h * 3600 + m * 60 + s + int(ms) / 1000.0
        return total_sec

    def draw_box(self, frame_face_data, img, thred, sea_epi, scale_factor):
        # if scale_factor != 1:
        #     img = self.resize_img(img, scale_factor)
        a = img.shape[1] / 1920

        font_size = max(int(50 * a), 15)  # Prevent font from being too small
        font = ImageFont.truetype(os.environ.get('EVAL_FONT_PATH', 'DejaVuSans.ttf'), font_size)
        text_bbox = []
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        speaker_path = os.path.join(self._get_drama_root(sea_epi), 'speaker_data')
        speak_map_path = None
        try:
            with open(os.path.join(speaker_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
                speaker_file = version_data['speaker_recognition_step']['active_version']
                speak_map_path = os.path.join(speaker_path, speaker_file, '.workspace/new_roles_map.json')
        except:
            pass

        if speak_map_path is not None:
            with open(speak_map_path) as f:
                mapping_name = json.load(f)
        else:
            mapping_name = {}

        episode = sea_epi.split('_')[1].zfill(4)
        for face in frame_face_data:
            role_name = face["id_list"]['0']['name']
            confidence = float(face["id_list"]['0']['score'])
            bbox = list(face["geometry"]['active_frame_list']['0']['box'].values())
            if confidence < thred or f'{episode}_pro' in role_name:
                continue
            if '少年' in role_name:
                role_name = '少年' + role_name.replace('少年','')
            elif '幼年' in role_name:
                role_name = '幼年' + role_name.replace('幼年', '')
            elif '童年' in role_name:
                role_name = '童年' + role_name.replace('童年', '')
            elif '成年' in role_name:
                role_name = '成年' + role_name.replace('成年', '')
            elif '青年' in role_name:
                role_name = '青年' + role_name.replace('青年', '')

            role_name = role_name.replace('（）','').replace('()','')
            if '新角色' in role_name:
                if role_name in mapping_name.keys():
                    role_name = mapping_name[role_name]
                else:
                    role_name = '其他'

            if role_name in replace_names.keys():
                # print(f'*****{role_name}***{replace_names[role_name]}******')
                role_name = replace_names[role_name]
            if 'New Role' in role_name or '新角色' in role_name:
                if role_name in mapping_name.keys():
                    role_name = mapping_name[role_name]
                else:
                    role_name = 'Others'

            left, top, right, bottom = font.getbbox(role_name)
            text_width = right - left
            text_height = bottom - top

            bbox = [max(0, _) * scale_factor  for _ in bbox]
            bbox[1] = max(text_height+5, bbox[1])
            bbox[2] = min(img.shape[1],  bbox[2])
            if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
                continue

            draw.rectangle(bbox, outline=(255, 0, 0), width=math.ceil(4*scale_factor))  # Here, a red line is used with a width of 2

            text_x = bbox[0] + (bbox[2] - bbox[0] - text_width) // 2
            text_y = bbox[1] - text_height - 5

            Bx1, By1, Bx2, By2 = text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5
            # Handle name box overlap
            for box_old in text_bbox:
                attemp = 0
                iou = self.calculate_iou(box_old, [Bx1, By1, Bx2, By2])
                while iou > 0.05 and attemp < 5:
                    By1 -= 20
                    By2 -= 20
                    text_y -= 20
                    attemp += 1
                    iou = self.calculate_iou(box_old, [Bx1, By1, Bx2, By2])

            # Draw text background rectangle
            draw.rectangle([(Bx1, By1), (Bx2, By2)], fill=(255, 0, 0))
            text_bbox.append([Bx1, By1, Bx2, By2])
            # Draw text
            draw.text((text_x, text_y), role_name, fill=(255, 255, 255), font=font)
        return np.array(pil_img)

    # Qwen-VL: patch_size(14) * spatial_merge_size(2) = 28, width and height must be
    # aligned to multiples of 28, otherwise image token placeholder count does not match
    # vision encoder output count, triggering masked_scatter device assertion.
    QWEN_VL_SIZE_FACTOR = 28

    def _align_to_factor(self, value, factor=None):
        factor = factor or self.QWEN_VL_SIZE_FACTOR
        return max(factor, int(round(value / factor)) * factor)

    def _align_pil_to_factor(self, pil_img, factor=None):
        # Align PIL image width and height to factor multiples to ensure image token
        # count matches placeholder count sent to vLLM.
        factor = factor or self.QWEN_VL_SIZE_FACTOR
        w, h = pil_img.size
        new_w = max(factor, int(round(w / factor)) * factor)
        new_h = max(factor, int(round(h / factor)) * factor)
        if (new_w, new_h) != (w, h):
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return pil_img

    def _target_size(self, w, h):
        """Scale images proportionally to target width, aligning width and height to
        multiples of 28. target_width mode: all episodes are resized to the same width
        (with different original resolutions: 1920x1080 / 1920x800 / 1280x720 ...),
        preserving aspect ratio, to avoid scaling issues where a 1280-wide episode is
        shrunk significantly compared to others when using scale_factor. When
        target_width is 0, revert to old scale_factor behavior."""
        if self.target_width:
            new_w = self._align_to_factor(self.target_width)
            new_h = self._align_to_factor(h * (self.target_width / float(w)))
        else:
            new_w = self._align_to_factor(w * self.scale_factor)
            new_h = self._align_to_factor(h * self.scale_factor)
        return new_w, new_h

    def resize_img_pil(self, img):
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img)

        new_size = self._target_size(pil_img.width, pil_img.height)

        # Scale (using high-quality resampling) or Image.BILINEAR
        resized_pil = pil_img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert back to numpy array
        resized_img = np.array(resized_pil)
        return resized_img

    def resize_img(self, img):
        w, h = img.size
        return img.resize(self._target_size(w, h), Image.Resampling.LANCZOS)

    def calculate_iou(self, box1, box2):
        """Compute the IoU (Intersection over Union) between two bounding boxes. Args:
        box1: [x1, y1, x2, y2] box2: [x1, y1, x2, y2] Returns: iou: float, Intersection
        over Union, in range [0, 1]"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate the coordinates of the intersection region
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        # Calculate the intersection area
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        # Calculate the individual areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Compute the union area
        union_area = area1 + area2 - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        iou = inter_area / union_area
        return iou

    def get_speaker(self, time, sample_res, has_season=False):
        speaker_save_info = {}
        for sample_item in sample_res:
            video_key = sample_item[0]
            episode = video_key.split('_')[1].zfill(4)
            season = video_key.split('_')[0]
            images = sample_item[1]
            frame_indices = sample_item[2]
            time_ = time.replace(':', '-')
            time_ = time_.replace('#', '-')
            speaker_path = os.path.join(self._get_drama_root(video_key), 'speaker_data')
            with open(os.path.join(speaker_path, 'version.json'), 'r', encoding='utf-8') as fr:
                version_data = json.load(fr)
                speaker_file = version_data['speaker_recognition_step']['active_version']
                speaker_file_path = os.path.join(speaker_path, speaker_file, episode+'.json')
                with open(speaker_file_path, 'r', encoding='utf-8') as file:
                    speaker_data = json.load(file)
                    for im_id in range(images.shape[0]):
                        frame_id = frame_indices[im_id]
                        if has_season == False:
                            frame_key = episode + '_' + str(frame_id)
                        else:
                            frame_key = season + '_' + episode + '_' + str(frame_id)
                        speaker_save_info[frame_key] = {}
                        for sub_key, info in speaker_data['object_list'].items():
                            start_frame = info['start_frame']
                            end_frame = info['end_frame']
                            if start_frame <= int(frame_id) <= end_frame:
                                speaker_save_info[frame_key] = {
                                    'role': info['role'],
                                    'text': info['text']
                                }

        sampled_speaker_path = os.path.join(self.image_path, time_, 'spkeaker.json')
        atomic_dump_json(speaker_save_info, sampled_speaker_path)

    def evaluation(self, result):
        gts = 0
        tps = 0
        total_tokens = 0
        missing_usage = 0
        skipped = 0
        no_extract = 0
        guessed = 0
        guessed_hit = 0
        for q_i, q_info in result.items():
            an = q_info['Q&A']['Answer']
            if 'result' not in q_info['Q&A'].keys():
                continue
            # Problems with missing video or missing image are not actually evaluated,
            # so they cannot be counted in the denominator; otherwise, they would be
            # incorrectly counted as wrong, lowering the accuracy.
            if q_info['Q&A'].get('skip_reason'):
                skipped += 1
                continue
            gts += 1
            pred = q_info['Q&A']['result']
            is_guess = bool(q_info['Q&A'].get('guessed'))
            if pred is None:
                no_extract += 1
            if is_guess:
                guessed += 1
            if pred == an:
                tps += 1
                if is_guess:
                    guessed_hit += 1
            reported_tokens = q_info['Q&A'].get('total_tokens')
            if reported_tokens is None:
                missing_usage += 1
            else:
                total_tokens += reported_tokens

        if gts == 0:
            print('model: {}, 无有效评测结果（跳过 {} 题）'.format(self.short_model_name, skipped))
            return

        acc = (tps / gts) * 100
        tokens_per_q = (f"unavailable ({missing_usage} responses omitted usage)"
                        if missing_usage else f"{total_tokens / gts:.2f}")
        print('model: {}, question num: {}, accuracy: {:.2f}, avg tokens per question {}'.format(
                                                                    self.short_model_name,
                                                                    gts, acc, tokens_per_q))
        print('  其中未能提取答案: {} 题 ({:.2f}%)，因缺数据跳过未计入: {} 题'.format(
            no_extract, no_extract / gts * 100, skipped))
        # Random fallback problems will "gain" some correct answers, which must be
        # reported separately; otherwise, the true performance cannot be judged.
        # real_acc is the lower bound after removing guessed correct answers.
        if guessed:
            real_acc = (tps - guessed_hit) / gts * 100
            print('  随机兜底: {} 题 ({:.2f}%)，其中猜中 {} 题；'
                  '剔除猜中后的准确率 {:.2f}%'.format(
                      guessed, guessed / gts * 100, guessed_hit, real_acc))

def main(args):
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                         type=str,
                         default='22002',
                         help='Another example parameter')
    parser.add_argument('--image_path',
                         type=str,
                         default='./results/img_data',
                         help='Another example parameter')
    parser.add_argument('--drama',
                         type=str,
                         default='a_lifelong_journey',
                         help='Another example parameter')
    parser.add_argument('--model_name',
                        type=str,
                        default="gpt-4.1",
                        help='[gpt-4.1, gpt-5, gpt-4o, gemini-2.5-pro, qwen2.5-vl-72b, qwen3-vl-32b,' \
                        'qwen3-vl-235b, glm4.6v, intern-vl-38b, intern-vl-241b, qwen2.5-vl-7b]')
    parser.add_argument('--model_path',
                        type=str,
                        default="",
                        help='model path')
    parser.add_argument('--question_path',
                        type=str,
                        default="",
                        help='question path')
    parser.add_argument('--sample_frames',
                         type=int,
                         default=64,
                         help='Another example parameter')
    parser.add_argument('--video_path',
                         type=str,
                         default='',
                         help='Another example parameter')
    parser.add_argument("--scale_factor",
                        type=float,
                        default=1,
                        help='Image scale factor; ignored when target width is set.')
    parser.add_argument("--target_width",
                        type=int,
                        default=0,
                        help='Resize frames to this width while preserving aspect ratio; 0 uses scale factor.')
    parser.add_argument('--subtitle_text',
                         action='store_true',
                         default=False,
                         help='Send text subtitles alongside frames when subtitles are not burned in.')
    parser.add_argument('--render_subtitle',
                         action='store_true',
                         default=True,
                         help='Burn subtitles into sampled frames (enabled by default).')
    parser.add_argument('--no_render_subtitle',
                         dest='render_subtitle',
                         action='store_false',
                         help='Disable subtitle burning.')
    parser.add_argument('--enable_thinking',
                        choices=['auto', 'on', 'off'],
                        default=os.environ.get('ENABLE_THINKING', 'auto'),
                        help='Thinking policy: auto follows the model template; on enables thinking; off disables it when supported.')
    parser.add_argument('--num_workers',
                         type=int,
                         default=16,
                         help='Another example parameter')
    parser.add_argument('--use_render_info',
                        action='store_true',
                        default=False,
                        help='Another example parameter')
    parser.add_argument('--use_rendered_video',
                        action='store_true',
                        default=False,
                        help='Read pre-rendered face videos instead of drawing face boxes.')
    parser.add_argument('--parallel',
                         action='store_true',
                         default=False,
                         help='Another example parameter')
    parser.add_argument('--media_resolution',
                         type=str,
                         default='low',
                         help='[low,medium,unspecified,high,ultra_high]')
    parser.add_argument('--level',
                         type=str,
                         default='L2',
                         help='Context window level; L6 covers the full drama, including all seasons.')
    parser.add_argument('--caption_path',
                         type=str,
                         default='',
                         help='Deprecated: QA records already contain context-window times.')
    parser.add_argument('--only_gen_img',
                         action='store_true',
                         default=False,
                         help='only sample imgage')
    parser.add_argument('--group_4x4',
                         action='store_true',
                         default=False,
                         help='group images in 4x4 grid')
    parser.add_argument('--group_2x2',
                         action='store_true',
                         default=False,
                         help='group images in 2x2 grid')
    parser.add_argument('--limit',
                         type=int,
                         default=0,
                         help='Legacy clip limit; 0 means all. Use dramavqa --limit for a question limit.')
    parser.add_argument('--shard_index',
                         type=int,
                         default=0,
                         help='Zero-based frame-generation shard index; requires --shard_total.')
    parser.add_argument('--shard_total',
                         type=int,
                         default=1,
                         help='Total frame-generation shards; 1 disables sharding.')
    parser.add_argument('--serving_mode',
                         type=str,
                         default='',
                         choices=['', 'local', 'api'],
                         help='Model source: local OpenAI-compatible server or external API; empty infers from model name.')
    parser.add_argument('--ablation_mode',
                         type=str,
                         default='',
                         choices=['', 'exp1_burn', 'exp2_nosub', 'exp3_gallery',
                                  'exp4_speaker', 'exp5_nosub', 'exp6_langonly'],
                         help='Ablation: burned subtitles, text subtitles, face gallery, speakers, frames only, or language only.')
    parser.add_argument('--speaker_root', type=str,
                        default=os.environ.get('SPEAKER_ROOT', ''),
                        help='Speaker metadata root containing <drama>/speaker_data/version.json.')
    parser.add_argument('--preflight_only', action='store_true',
                        help='Validate speaker-mode frame, subtitle and speaker inputs without inference.')
    parser.add_argument('--result_path',
                         type=str,
                         default='',
                         help='Another example parameter')


    args = parser.parse_args(args)
    if args.preflight_only and (args.only_gen_img or
            (args.ablation_mode or os.environ.get('ABLATION_MODE')) != 'exp4_speaker'):
        parser.error('--preflight_only 仅用于 exp4_speaker 推理输入检查')
    try:
        engine = LLMInference(args)
    except NoVideoAvailable as e:
        # Missing data is not a program error: normal exit, let start.sh continue
        # running the rest of the plays
        print(f'[SKIP] {e}')
        return
    # read_caption_info() is deprecated: QA data already contains time information at
    # each level
    # engine.read_caption_info()
    engine.read_question_info()
    if args.preflight_only:
        return

    if args.only_gen_img == False:
        time1 = time.time()
        if args.parallel:
            result = engine.parallel_inference()
        else:
            result = engine.single_inference()
        time2 = time.time()
        print(f'cost: {time2 - time1} s')
        engine.evaluation(result)

if __name__ == "__main__":
    main(sys.argv[1:])
