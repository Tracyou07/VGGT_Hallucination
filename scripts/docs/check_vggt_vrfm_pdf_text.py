from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from build_vggt_vrfm_pdfs import DOCUMENTS, ROOT


def extract_text(pdf_path: Path) -> str:
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("required executable is unavailable: pdftotext")
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf_path), "-"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"pdftotext failed for {pdf_path.name}: {error}")
    return result.stdout.decode("utf-8", errors="replace")


def check_document(key: str) -> list[str]:
    document = DOCUMENTS[key]
    pdf_path = ROOT / document.output
    if not pdf_path.is_file():
        return [f"{key}: PDF does not exist: {document.output}"]

    text = extract_text(pdf_path)
    visible = "".join(text.split())
    errors: list[str] = []
    if len(visible) < 100:
        errors.append(f"{key}: extracted PDF text is unexpectedly short")
    if "�" in text:
        errors.append(f"{key}: extracted PDF text contains replacement characters")
    if "??" in text:
        errors.append(f"{key}: PDF contains an unresolved-reference marker (??)")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check mechanical text quality of the VGGT research PDFs."
    )
    parser.add_argument(
        "--document",
        choices=[*DOCUMENTS, "all"],
        default="all",
        help="document to inspect (default: all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list public PDF outputs without checking them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for document in DOCUMENTS.values():
            print(f"{document.key}\t{document.output}")
        return 0

    keys = DOCUMENTS if args.document == "all" else (args.document,)
    failures: list[str] = []
    try:
        for key in keys:
            failures.extend(check_document(key))
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if failures:
        for failure in failures:
            print(f"error: {failure}", file=sys.stderr)
        return 1

    for key in keys:
        print(f"checked\t{key}\t{DOCUMENTS[key].output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

