"""Resolve current flat media directories and historical versioned downloads."""
from pathlib import Path
import json


def resolve_data_dir(parent_dir, step_key, fallback_keyword=None):
    """Prefer flat episode files; retain optional legacy version manifests."""
    parent = Path(parent_dir)
    if not parent.is_dir():
        return None
    if any(p.is_file() and p.stem.isdigit() and p.suffix.lower() in
           {".json", ".mp4", ".mkv"} for p in parent.iterdir()):
        return str(parent)
    try:
        versions = json.loads((parent / "version.json").read_text(encoding="utf-8"))
        steps = (step_key,) if isinstance(step_key, str) else step_key
        for step in steps:
            active = (versions.get(step) or {}).get("active_version")
            if active and (parent / active).is_dir():
                return str(parent / active)
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    subdirs = sorted(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if fallback_keyword:
        matching = [p for p in subdirs if fallback_keyword in p.name]
        if matching:
            return str(matching[-1])
    return str(subdirs[-1]) if subdirs else None
