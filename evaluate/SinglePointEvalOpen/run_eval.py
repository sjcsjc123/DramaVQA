"""Compatibility convenience entry point; prefer dramavqa run --task single-point."""

import sys
from dramavqa.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", "--task", "single-point", *sys.argv[1:]]))
