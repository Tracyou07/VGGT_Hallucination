from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Document:
    key: str
    source: str
    output: str


DOCUMENTS = {
    document.key: document
    for document in (
        Document(
            "theory",
            "doc/camera_solution_space_01_theory_foundation/"
            "camera_trajectory_solution_space.tex",
            "output/pdf/camera_trajectory_solution_space_theory.pdf",
        ),
        Document(
            "experiment",
            "doc/camera_velocity_ambiguity_preexperiment/"
            "camera_velocity_ambiguity_preexperiment.tex",
            "output/pdf/camera_velocity_ambiguity_preexperiment.pdf",
        ),
        Document(
            "method",
            "doc/variational_rectified_camera_refiner/"
            "variational_rectified_camera_refiner_method.tex",
            "output/pdf/variational_rectified_camera_refiner_method.pdf",
        ),
    )
}


def run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        sys.stdout.write(result.stdout)
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            + " ".join(command)
        )


def require_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"required executable is unavailable: {name}")
    return executable


def build_document(document: Document, render: bool) -> None:
    source = ROOT / document.source
    if not source.is_file():
        raise RuntimeError(f"TeX source does not exist: {document.source}")

    xelatex = require_executable("xelatex")
    bibtex = require_executable("bibtex")
    build_dir = ROOT / "tmp" / "pdfs" / document.key
    build_dir.mkdir(parents=True, exist_ok=True)

    latex_command = [
        xelatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-output-directory={build_dir}",
        str(source),
    ]
    run(latex_command)
    run([bibtex, str(build_dir / source.stem)])
    run(latex_command)
    run(latex_command)

    built_pdf = build_dir / f"{source.stem}.pdf"
    output_pdf = ROOT / document.output
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_pdf, output_pdf)
    print(f"built\t{document.key}\t{document.output}")

    if render:
        pdftoppm = require_executable("pdftoppm")
        render_dir = ROOT / "tmp" / "pdf-renders" / document.key
        render_dir.mkdir(parents=True, exist_ok=True)
        for old_page in render_dir.glob("page-*.png"):
            old_page.unlink()
        run(
            [
                pdftoppm,
                "-png",
                "-r",
                "144",
                str(output_pdf),
                str(render_dir / "page"),
            ]
        )
        print(f"rendered\t{document.key}\t{render_dir.relative_to(ROOT).as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the VGGT repair-velocity research PDF set."
    )
    parser.add_argument(
        "--document",
        choices=[*DOCUMENTS, "all"],
        default="all",
        help="document to build (default: all)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render every built PDF page to PNG with Poppler",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list public source/output mappings without building",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for document in DOCUMENTS.values():
            print(f"{document.key}\t{document.source}\t{document.output}")
        return 0

    selected = (
        DOCUMENTS.values()
        if args.document == "all"
        else (DOCUMENTS[args.document],)
    )
    try:
        for document in selected:
            build_document(document, render=args.render)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

