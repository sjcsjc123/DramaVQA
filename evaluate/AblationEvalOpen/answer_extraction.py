"""Ground-truth-independent final-choice extraction for ablation evaluation.

Explicit formats shared with SingleHopEval/MultiPointEval are accepted, but
never search arbitrary prose for a bracketed candidate. Reasoning fallback
requires a terminal explicit answer marker (not a bare reasoning tail).
"""
import re

VERSION = 'aligned_terminal_options_v3'


def extract_final_choice(text, options=None, *, reasoning_only=False):
    if not isinstance(text, str) or not text.strip():
        return None
    allowed = None
    if options is not None:
        allowed = set(re.findall(r'^\s*\**\s*([A-Z])\s*[.、,，):：]', options, re.M))
        if not allowed:
            return None
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail = lines[-1]
    tail = re.sub(r'^#{1,6}\s*', '', tail)
    explicit = [
        r'(?:Final\s*Answer|Answer|答案)\s*[:：]\s*\**\s*[<\[]?\s*([A-Za-z])\s*[>\]]?\s*[.、,，)]?\s*\**',
        r'<\|begin_of_box\|>\s*([A-Za-z])\s*<\|end_of_box\|>',
        r'<(?:letter|answer|option|choice|ref|ans)>\s*\**\s*([A-Za-z])\s*\**\s*</(?:letter|answer|option|choice|ref|ans)>',
    ]
    # Strip presentation-only bold around the entire conclusion.
    tail = tail.strip('*').strip()
    patterns = list(explicit)
    if not reasoning_only:
        patterns += [r'<\s*([A-Za-z])\s*>?', r'\[\s*([A-Za-z])\s*\]',
                     r'\**\s*([A-Za-z])\s*[.、,，)]?\s*\**']
        if len(lines) == 1:
            patterns += [r'<code>\s*([A-Za-z])\s*</code>',
                         r'<(?:b|\\b)>\s*([A-Za-z])\s*<(?:/b|\\b)>',
                         r'<\\[bnt]>\s*([A-Za-z])',
                         r'<(?:\\n|\\t|\s)*([A-Za-z])(?:\\n|\\t|\s)*>?',
                         r'<letter>\s*([A-Za-z])']
    for pattern in patterns:
        match = re.fullmatch(pattern, tail, re.I)
        if match:
            letter = match.group(1).upper()
            return letter if allowed is None or letter in allowed else None
    # Exact option echo only: do not accept arbitrary prose after a letter.
    if not reasoning_only and isinstance(options, str):
        match = re.fullmatch(r'<?\s*([A-Za-z])\s*[.、,，):：]\s*([^<>]+)>?', tail)
        if match:
            letter, content = match.group(1).upper(), match.group(2)
            candidates = re.findall(r'^\s*' + letter + r'\s*[.、,，):：]\s*(.+)$', options, re.M)
            normalize = lambda value: ' '.join(value.split())
            if len(candidates) == 1 and normalize(content) == normalize(candidates[0]):
                return letter
    return None
