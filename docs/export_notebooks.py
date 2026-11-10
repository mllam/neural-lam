"""Export marimo notebook sources to Markdown before building with Sphinx."""

# Standard library
import argparse
import subprocess
import sys
from pathlib import Path

# Notebooks executed at docs build time, so their pages show cell outputs
EXECUTED_NOTEBOOKS = ("hello_world_danra",)

RENDERED_PAGE = """\
# {title}

<iframe src="../_static/notebooks/{name}.html" title="{name}"
  style="width: 100%; height: 85vh; border: none;"></iframe>

<p><a href="../_static/notebooks/{name}.html">Open the rendered notebook
in a full page</a></p>
"""


def first_heading(markdown: str) -> str:
    """Return the first level-one heading outside of fenced code blocks."""
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith("# "):
            return line.removeprefix("# ")
    raise ValueError("Notebook has no level-one heading")


def main() -> None:
    """Export each notebook to Markdown, executing the selected ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Export every notebook without executing it (e.g. linkcheck)",
    )
    args = parser.parse_args()

    docs_dir = Path(__file__).resolve().parent
    notebook_dir = docs_dir / "notebooks"
    rendered_dir = docs_dir / "_static" / "notebooks"
    notebooks = sorted(notebook_dir.glob("*.py"))
    # Markdown export does not validate cells, so fail on broken notebooks here
    subprocess.run(
        [sys.executable, "-m", "marimo", "check", *map(str, notebooks)],
        check=True,
    )
    for notebook in notebooks:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "marimo",
                "export",
                "md",
                str(notebook),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        page = result.stdout
        if notebook.stem in EXECUTED_NOTEBOOKS and not args.no_execute:
            # Runs every cell and fails if any of them raises
            rendered_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "html",
                    "--force",
                    str(notebook),
                    "-o",
                    str(rendered_dir / f"{notebook.stem}.html"),
                ],
                check=True,
            )
            page = RENDERED_PAGE.format(
                title=first_heading(page), name=notebook.stem
            )
        # Replace generated pages only after a successful export. Using stdout
        # also avoids overwrite prompts when rebuilding the documentation.
        notebook.with_suffix(".md").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
