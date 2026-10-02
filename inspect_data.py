#!/usr/bin/env python3
"""Print the format, column names, and sample ids of XRD and synthesis files.

Examples
--------
python inspect_data.py
python inspect_data.py --xrd-dir data/xrd --synthesis data/synthesis_params.csv
python inspect_data.py --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from demo_data import DEMO_DIR, ensure_demo_dataset
from loader import format_inspection_report, inspect_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect XRD spectra and synthesis-parameter files.")
    parser.add_argument("--xrd-dir", type=Path, default=Path("data/xrd"))
    parser.add_argument("--synthesis", type=Path, default=Path("data/synthesis_params.csv"))
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Inspect the synthetic demo dataset instead of data/xrd",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xrd_dir = args.xrd_dir
    synthesis = args.synthesis
    if args.demo:
        ensure_demo_dataset(DEMO_DIR)
        xrd_dir = DEMO_DIR / "xrd"
        synthesis = DEMO_DIR / "synthesis_params.csv"
        print("Inspecting the SYNTHETIC demo dataset. These files are not experimental measurements.\n")
    elif not xrd_dir.exists() or not any(xrd_dir.rglob("*")):
        print(
            "No experimental XRD files were found. "
            "Add patterns under data/xrd, or rerun with --demo to inspect the synthetic example.\n"
        )
    report = inspect_inputs(xrd_dir, synthesis)
    print(format_inspection_report(report))


if __name__ == "__main__":
    main()
