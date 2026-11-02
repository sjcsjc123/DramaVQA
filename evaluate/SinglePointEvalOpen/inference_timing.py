"""Non-invasive, per-question timing for the compatibility evaluator.

One event covers get_llm_response(): input preparation has already finished;
API retries, rate-limit waits and answer-extraction retries are included.
Instrumentation never changes the returned answer or exception. Events go to
answer_timings.jsonl alongside raw_responses.jsonl, not into q_a_dict.json.
"""
import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

SCHEMA = 'answer_cycle_v1'


def question_fingerprint(qa):
    """Stable across incremental key renumbering; never hash the ground truth."""
    question = next((qa[k].strip() for k in ('Question', 'question', 'Q')
                     if isinstance(qa.get(k), str) and qa[k].strip()), None)
    if question is None:
        return None
    payload = json.dumps([question, qa.get('Options')], ensure_ascii=False,
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _append_event(engine, key, qa, started_at, elapsed_s, outcome, error):
    raw_path = getattr(engine, '_raw_log_path', None)
    if not raw_path:
        return
    answer, skip = (outcome[0], outcome[3]) if outcome is not None else (None, None)
    status = ('exception' if error else 'api_failure' if skip else
              'extraction_failed' if answer is None else 'answered')
    record = {
        'schema': SCHEMA,
        'event_id': uuid.uuid4().hex,
        'started_at': started_at,
        'finished_at': datetime.datetime.now().astimezone().isoformat(timespec='microseconds'),
        'elapsed_s': elapsed_s,
        'key': str(key),
        'question_fingerprint': question_fingerprint(qa or {}),
        'model': engine.model_name,
        'result_model': getattr(engine, 'short_model_name', engine.model_name),
        'serving_mode': getattr(engine, 'serving_mode', None),
        'drama': engine.drama,
        'level': engine.level,
        'sample_frames': engine.sample_frames,
        'question_path': getattr(engine, 'question_path', ''),
        'num_workers': getattr(engine, 'num_workers', None),
        'parallel': getattr(engine, 'parallel', None),
        'status': status,
        'result': answer,
        'skip_reason': skip,
        'error_type': type(error).__name__ if error else None,
        'pid': os.getpid(),
    }
    # Reuse the existing per-engine lock: concurrent calls must not interleave
    # their JSON lines, and all calls in one engine share the same run id.
    with engine._raw_log_lock:
        if not getattr(engine, '_timing_run_id', None):
            engine._timing_run_id = uuid.uuid4().hex
        record['run_id'] = engine._timing_run_id
        with Path(raw_path).with_name('answer_timings.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def measure_answer_time(function):
    """Wrap the existing four-value response function without changing policy."""
    @functools.wraps(function)
    def wrapped(self, msg, key, *args, **kwargs):
        started_at = datetime.datetime.now().astimezone().isoformat(timespec='microseconds')
        start = time.perf_counter()
        outcome = None
        error = None
        try:
            outcome = function(self, msg, key, *args, **kwargs)
            return outcome
        except BaseException as exc:
            error = exc
            raise
        finally:
            elapsed_s = time.perf_counter() - start
            try:
                _append_event(self, key, kwargs.get('qa'), started_at, elapsed_s, outcome, error)
            except Exception as exc:
                # Missing metadata, disk-full or logging errors cannot turn a
                # successful inference into a failure or mask its real exception.
                try:
                    print(f'[warn] answer_timings.jsonl 写入失败: {type(exc).__name__}: {exc}')
                except Exception:
                    # Closed stdout / a broken log pipe must not mask the
                    # inference return value or the original exception either.
                    pass
    return wrapped
