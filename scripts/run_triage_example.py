"""CLI helper to exercise the Hackathon triage pipeline.

Usage:
    python scripts/run_triage_example.py --input examples/hackathon_triage_input.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from hackathon import NursingTriageInput, example_triage_payload, generate_triage_report


def load_payload(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return example_triage_payload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Runs the Hackathon triage pipeline.")
    parser.add_argument(
        "--input",
        default="examples/hackathon_triage_input.json",
        help="JSON with data collected by the nursing staff.",
    )
    parser.add_argument(
        "--output",
        default="examples/hackathon_triage_output.json",
        help="File where the structured report will be saved.",
    )
    args = parser.parse_args()

    input_path = ROOT / args.input
    output_path = ROOT / args.output

    payload = load_payload(input_path)
    triage_input = NursingTriageInput(**payload)
    report = generate_triage_report(triage_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Triage summary:")
    print(report.summary_markdown())
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
