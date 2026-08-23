#!/usr/bin/env python3
"""Seal an immutable decoded fixed-eight observation from a plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pre_experiments.camera_solution_space_01.observation import seal_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan_json")
    parser.add_argument("source_sens")
    parser.add_argument("output_parent")
    arguments = parser.parse_args()
    plan = json.loads(Path(arguments.plan_json).read_text(encoding="utf-8"))
    print(seal_observation(plan, arguments.source_sens, arguments.output_parent))


if __name__ == "__main__":
    main()
