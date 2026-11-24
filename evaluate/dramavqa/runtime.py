"""Explicit evaluation settings and a bounded subprocess environment."""
import argparse
import os


# Keep OS/runtime connectivity and credentials, never inherited experiment controls.
_RUNTIME_ENV = {
    'PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'LANG', 'LANGUAGE', 'LC_ALL', 'TZ',
    'TMP', 'TEMP', 'TMPDIR', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT',
    'SYSTEMDRIVE', 'USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'PROGRAMDATA',
    'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMW6432', 'OS',
    'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE',
    'PYTHONPATH', 'PYTHONUTF8', 'PYTHONIOENCODING', 'PYTHONUNBUFFERED',
    'VIRTUAL_ENV', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'LD_LIBRARY_PATH',
    'CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
    'NVIDIA_DRIVER_CAPABILITIES', 'HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE',
    'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 'XDG_CACHE_HOME',
    'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
    'OPENAI_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
}


def child_environment(source=None):
    source = os.environ if source is None else source
    return {key: value for key, value in source.items()
            if key.upper() in _RUNTIME_ENV or key.upper().startswith('LC_')}


def add_execution_arguments(parser):
    parser.add_argument('--max-retries', type=int, default=6, help='Maximum API attempts per request (default: 6)')
    parser.add_argument('--tpm-limit', type=int, default=0, help='Token-per-minute quota; 0 disables pacing')
    parser.add_argument('--decode-threads', type=int, default=2, help='Video decoder threads (default: 2)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (default: 0.1 for API, 1 for Conan)')
    parser.add_argument('--top-p', type=float, default=0.01, help='Nucleus sampling probability (default: 0.01)')
    parser.add_argument('--reasoning-effort', choices=('low', 'high', 'max'), help='Explicit API reasoning effort; omitted by default')
    parser.add_argument('--thinking-token-budget', type=int, help='Single-point API thinking token budget; omitted by default')
    parser.add_argument('--guess-after-failed-retries', type=int, help='Single-point strict policy: failed runs before a marked random fallback')
    parser.add_argument('--generation-policy', choices=('baseline', 'strict'), default='baseline',
                        help='Single-point extraction policy; strict keeps extraction failures pending')
    parser.add_argument('--run-tag', default='', help='Label recorded with this run')
    parser.add_argument('--retry-guessed', action=argparse.BooleanOptionalAction, default=True,
                        help='Retry previously guessed single-point answers (default: enabled)')
    parser.add_argument('--restrict-to-question-set', action=argparse.BooleanOptionalAction, default=True,
                        help='Restrict single-point resume to the current QA file (default: enabled)')


def execution_settings(args):
    return {name: getattr(args, name) for name in (
        'max_retries', 'tpm_limit', 'decode_threads', 'temperature', 'top_p',
        'reasoning_effort', 'thinking_token_budget', 'guess_after_failed_retries',
        'generation_policy', 'run_tag', 'retry_guessed', 'restrict_to_question_set')}


def execution_arguments(args):
    result = []
    for name, value in execution_settings(args).items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            result.append(flag if value else '--no-' + name.replace('_', '-'))
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def validate_execution_settings(args):
    if args.max_retries < 1 or args.tpm_limit < 0 or args.decode_threads < 1:
        raise ValueError('max-retries/decode-threads must be positive; tpm-limit nonnegative')
    if not 0 <= args.temperature <= 2 or not 0 < args.top_p <= 1:
        raise ValueError('temperature must be in [0, 2]; top-p in (0, 1]')
    if args.thinking_token_budget is not None and not 0 < args.thinking_token_budget <= args.max_tokens - 256:
        raise ValueError('thinking-token-budget must be positive and leave 256 output tokens')
    if args.guess_after_failed_retries is not None and args.guess_after_failed_retries < 1:
        raise ValueError('guess-after-failed-retries must be positive')
    if args.guess_after_failed_retries is not None and args.generation_policy != 'strict':
        raise ValueError('guess-after-failed-retries requires --generation-policy strict')
    if args.max_tokens < 1 or not 0 < args.api_timeout <= 86400:
        raise ValueError('max-tokens must be positive; api-timeout must be in (0, 86400]')


def validate_task_settings(args, *, multi=False):
    if multi and (args.thinking_token_budget is not None
                  or args.guess_after_failed_retries is not None
                  or args.generation_policy != 'baseline'
                  or not args.retry_guessed or not args.restrict_to_question_set):
        raise ValueError('Single-point generation/resume policies are not supported for multi-scene evaluation')
    if getattr(args, 'backend', 'openai') == 'conan' and (
            args.max_retries != 6 or args.tpm_limit != 0 or args.top_p != 0.01
            or args.reasoning_effort is not None or args.thinking_token_budget is not None
            or args.guess_after_failed_retries is not None or args.generation_policy != 'baseline'
            or not args.retry_guessed or not args.restrict_to_question_set):
        raise ValueError('API generation/resume policies are not supported by the Conan launcher')
