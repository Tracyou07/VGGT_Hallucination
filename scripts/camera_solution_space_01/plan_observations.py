#!/usr/bin/env python3
"""Create a canonical fixed-eight observation plan from explicit eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pre_experiments.camera_solution_space_01.contracts import canonical_json_bytes
from pre_experiments.camera_solution_space_01.observation import plan_observation
from pre_experiments.camera_solution_space_01.sens_index import index_sens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_sens")
    parser.add_argument("scene_id")
    parser.add_argument("split")
    parser.add_argument("eligibility_json")
    parser.add_argument("output_plan")
    arguments = parser.parse_args()
    eligibility = json.loads(Path(arguments.eligibility_json).read_text(encoding="utf-8"))
    explicit = {int(key): value for key, value in eligibility.items()}
    plan = plan_observation(arguments.source_sens, index_sens(arguments.source_sens), arguments.scene_id, arguments.split, explicit)
    Path(arguments.output_plan).write_bytes(canonical_json_bytes(plan))
    print(plan["plan_id"])


if __name__ == "__main__":
    main()
