#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    path = parser.parse_args().path
    source = path.read_text()
    anchor = '                "separate_hook",\n'
    if anchor not in source:
        raise RuntimeError("ddp_comm_hook choices anchor not found")
    additions = ""
    for name in ("fp8_granularity_hook", "fp8_per_tensor_hook"):
        if f'"{name}"' not in source:
            additions += f'                "{name}",\n'
    if additions:
        source = source.replace(anchor, anchor + additions, 1)
        path.write_text(source)


if __name__ == "__main__":
    main()
