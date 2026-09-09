#!/usr/bin/env python3
"""Add experiment-only FP8 hook names to the train argument allowlist."""
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    path = parser.parse_args().path
    source = path.read_text()
    missing = []
    if '"fp8_granularity_hook"' not in source:
        missing.append('                "fp8_granularity_hook",\n')
    if '"fp8_per_tensor_hook"' not in source:
        missing.append('                "fp8_per_tensor_hook",\n')
    if missing:
        anchor = '                "separate_hook",\n'
        if anchor not in source:
            raise RuntimeError("ddp_comm_hook choices anchor not found")
        source = source.replace(anchor, anchor + "".join(missing), 1)
        path.write_text(source)


if __name__ == "__main__":
    main()
