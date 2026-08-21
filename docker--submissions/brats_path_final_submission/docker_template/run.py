#!/usr/bin/env python3
"""Challenge entrypoint template.

Replace the implementation in src/inference.py with your model loading and
prediction logic. Keep the CLI contract stable: --input and --output.

ADAPT: For most algorithms, leave this file unchanged and replace
src/inference.py. If you rename the inference module/function, update the import
below while preserving the --input/--output interface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ADAPT: Update only if your inference entrypoint is renamed.
from src.inference import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run challenge inference.")
    # ADAPT: Keep these names unless the target evaluator requires different
    # container arguments.
    parser.add_argument(
        "--input", required=True, help="Input directory inside the container."
    )
    parser.add_argument(
        "--output", required=True, help="Output directory inside the container."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print("[run] Starting challenge inference.", flush=True)
    print(f"[run] Input directory: {input_dir}", flush=True)
    print(f"[run] Output directory: {output_dir}", flush=True)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] Output directory is ready: {output_dir}", flush=True)

    run_inference(input_dir=input_dir, output_dir=output_dir)
    print("[run] Finished challenge inference.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
