"""Check lightweight runtime dependencies for AutoDL scripts."""

from __future__ import annotations

import argparse
import importlib.util
import sys


REQUIRED_MODULES: list[tuple[str, str]] = [
    ("cv2", "opencv-python-headless==4.11.0.86"),
    ("imageio", "imageio"),
    ("PIL", "Pillow"),
    ("plyfile", "plyfile"),
    ("scipy", "scipy"),
    ("safetensors", "safetensors"),
    ("einops", "einops"),
]


def missing_package_specs(required: list[tuple[str, str]] | None = None) -> list[str]:
    missing: list[str] = []
    for module_name, package_spec in required or REQUIRED_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_spec)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-missing-packages", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = missing_package_specs()
    if args.print_missing_packages:
        print(" ".join(missing))
        return
    if missing:
        print("[deps] missing runtime modules for packages:", " ".join(missing), file=sys.stderr)
        raise SystemExit(42)
    print("[deps] runtime dependency import checks passed")


if __name__ == "__main__":
    main()
