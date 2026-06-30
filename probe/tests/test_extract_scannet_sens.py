import importlib.util
import unittest
from pathlib import Path


def load_extract_module():
    path = Path("scripts/autodl/extract_scannet_sens.py").resolve()
    spec = importlib.util.spec_from_file_location("extract_scannet_sens", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExtractScanNetSensTest(unittest.TestCase):
    def test_default_process_dir_uses_scannet_root(self):
        extract = load_extract_module()
        raw_dir = Path("/root/autodl-tmp/datasets/scannetv2/raw_sens/scans")

        out_dir = extract.default_process_dir(raw_dir)

        self.assertEqual(out_dir, Path("/root/autodl-tmp/datasets/scannetv2/process_scannet"))


if __name__ == "__main__":
    unittest.main()
