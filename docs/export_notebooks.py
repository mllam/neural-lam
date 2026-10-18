"""Export marimo notebook sources to Markdown before building with Sphinx."""

# Standard library
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Export each notebook to Markdown without executing it."""
    notebook_dir = Path(__file__).resolve().parent / "notebooks"
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
        # Replace generated pages only after a successful export. Using stdout
        # also avoids overwrite prompts when rebuilding the documentation.
        notebook.with_suffix(".md").write_text(result.stdout, encoding="utf-8")


if __name__ == "__main__":
    main()
