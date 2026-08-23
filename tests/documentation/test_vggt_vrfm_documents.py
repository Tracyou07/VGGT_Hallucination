from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "docs" / "build_vggt_vrfm_pdfs.py"
CHECK_SCRIPT = ROOT / "scripts" / "docs" / "check_vggt_vrfm_pdf_text.py"

EXPECTED_DOCUMENTS = [
    (
        "theory",
        "doc/camera_solution_space_01_theory_foundation/"
        "camera_trajectory_solution_space.tex",
        "output/pdf/camera_trajectory_solution_space_theory.pdf",
    ),
    (
        "experiment",
        "doc/camera_velocity_ambiguity_preexperiment/"
        "camera_velocity_ambiguity_preexperiment.tex",
        "output/pdf/camera_velocity_ambiguity_preexperiment.pdf",
    ),
    (
        "method",
        "doc/variational_rectified_camera_refiner/"
        "variational_rectified_camera_refiner_method.tex",
        "output/pdf/variational_rectified_camera_refiner_method.pdf",
    ),
]


def run_script(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(path), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


class DocumentCommandLineContractTest(unittest.TestCase):
    def test_build_list_reports_the_three_public_artifacts(self):
        result = run_script(BUILD_SCRIPT, "--list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            ["\t".join(document) for document in EXPECTED_DOCUMENTS],
        )

    def test_text_check_list_uses_the_same_pdf_outputs(self):
        result = run_script(CHECK_SCRIPT, "--list")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [f"{key}\t{output}" for key, _, output in EXPECTED_DOCUMENTS],
        )

    def test_build_rejects_an_unknown_document_key(self):
        result = run_script(BUILD_SCRIPT, "--document", "unknown")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
