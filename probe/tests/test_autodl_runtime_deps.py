import importlib.util
import unittest
from pathlib import Path


def load_deps_module():
    path = Path("scripts/autodl/check_runtime_deps.py").resolve()
    spec = importlib.util.spec_from_file_location("check_runtime_deps", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AutoDLRuntimeDepsTest(unittest.TestCase):
    def test_missing_package_specs_maps_modules_to_packages(self):
        deps = load_deps_module()

        missing = deps.missing_package_specs(
            [("__definitely_missing_vggt_module__", "example-package==1.0")]
        )

        self.assertEqual(missing, ["example-package==1.0"])


if __name__ == "__main__":
    unittest.main()
