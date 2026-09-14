"""Plotting helpers for figure styling and LaTeX detection."""

# Standard library
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

# Third-party
from tueplots import bundles, figsizes


@cache
def has_working_latex() -> bool:
    """
    Check whether a LaTeX toolchain is available on the system.

    Returns
    -------
    bool
        ``True`` if ``latex`` and the required auxiliary tools are callable.
    """
    # If latex/toolchain is not available, some visualizations might not render
    # correctly, but will at least not raise an error. Alternatively, use
    # unicode raised numbers.

    if not shutil.which("latex"):
        return False
    if not shutil.which("dvipng"):
        return False
    if not (
        shutil.which("gs")
        or shutil.which("gswin64c")
        or shutil.which("gswin32c")
    ):
        return False

    tex_src = r"""
\documentclass{article}
\usepackage{fix-cm}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\begin{document}
$E=mc^2$ \LaTeX\ ok
\end{document}
""".lstrip()

    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "test.tex").write_text(tex_src, encoding="utf-8")
            cmd = [
                "latex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "test.tex",
            ]
            subprocess.run(
                cmd,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=True,
            )
            cmd_dvipng = ["dvipng", "-D", "100", "-o", "test.png", "test.dvi"]
            subprocess.run(
                cmd_dvipng,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=20,
                check=True,
            )
        return True
    except Exception:
        return False


def fractional_plot_bundle(fraction: float) -> dict[str, Any]:
    """
    Return a ``tueplots`` bundle scaled to a fraction of the page width.

    Parameters
    ----------
    fraction : float
        Denominator applied to the default NeurIPS figure width.

    Returns
    -------
    dict
        Matplotlib rcParams bundle with updated ``figure.figsize``.
    """
    usetex = has_working_latex()
    bundle = bundles.neurips2023(usetex=usetex, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] * fraction,
        original_figsize[1] * fraction,
    )
    return bundle
