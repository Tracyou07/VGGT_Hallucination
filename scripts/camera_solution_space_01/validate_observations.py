#!/usr/bin/env python3
"""Deeply validate a sealed fixed-eight observation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pre_experiments.camera_solution_space_01.observation import validate_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_root")
    parser.add_argument("source_sens")
    arguments = parser.parse_args()
    print(validate_observation(arguments.observation_root, arguments.source_sens))


if __name__ == "__main__":
    main()
