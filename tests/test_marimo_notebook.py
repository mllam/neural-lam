"""Check the dataset tutorial with small, deterministic input arrays."""

# Standard library
import shutil
import subprocess
import sys
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "docs/notebooks/create_reduced_meps_dataset.py"
)


def test_reduce_meps_notebook(tmp_path):
    """Execute every processing cell and check all three sample formats."""
    pytest.importorskip("marimo", reason="Requires the docs extra")
    source = tmp_path / "source"
    output = tmp_path / "reduced"
    static = source / "static"
    static.mkdir(parents=True)
    nx, ny, nvars = 48, 52, 10
    crop = (slice(12, 36), slice(13, 39))
    grid = np.arange(2 * nx * ny).reshape(2, nx, ny)
    surface = np.arange(nx * ny).reshape(nx, ny)
    border = np.ones((nx, ny), dtype=bool)
    border[10:-10, 10:-10] = False
    np.save(static / "nwp_xy.npy", grid)
    np.save(static / "surface_geopotential.npy", surface)
    np.save(static / "border_mask.npy", border)
    np.save(static / "parameter_weights.npy", np.arange(nvars))
    grid_features = torch.arange(nx * ny * 4).reshape(nx * ny, 4)
    torch.save(grid_features, static / "grid_features.pt")
    flux_stats = torch.tensor([1.0, 2.0])
    torch.save(flux_stats, static / "flux_stats.pt")
    stats = ("diff_mean", "diff_std", "parameter_mean", "parameter_std")
    for name in stats:
        torch.save(torch.arange(nvars), static / f"{name}.pt")

    state = np.arange(2 * nx * ny * nvars).reshape(2, nx, ny, nvars)
    flux = np.arange(2 * nx * ny).reshape(2, nx, ny)
    samples = {
        "nwp_2022040100_mbr000.npy": (state, state[:, crop[0], crop[1], :8]),
        "wtr_2022040100.npy": (surface, surface[crop]),
        "nwp_toa_downwelling_shortwave_flux_2022040112.npy": (
            flux,
            flux[:, crop[0], crop[1]],
        ),
    }
    for split in ("train", "test", "val"):
        directory = source / "samples" / split
        directory.mkdir(parents=True)
        for name, (original, _) in samples.items():
            np.save(directory / name, original)

    # Repeat the run to check the documented overwrite behavior as well.
    for _ in range(2):
        subprocess.run(
            [
                sys.executable,
                str(NOTEBOOK),
                "--input-dir",
                str(source),
                "--output-dir",
                str(output),
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        result_static = output / "static"
        np.testing.assert_array_equal(
            np.load(result_static / "nwp_xy.npy"), grid[:, crop[0], crop[1]]
        )
        np.testing.assert_array_equal(
            np.load(result_static / "surface_geopotential.npy"), surface[crop]
        )
        expected_border = np.ones((24, 26), dtype=bool)
        expected_border[10:-10, 10:-10] = False
        np.testing.assert_array_equal(
            np.load(result_static / "border_mask.npy"), expected_border
        )
        np.testing.assert_array_equal(
            np.load(result_static / "parameter_weights.npy"), np.arange(8)
        )
        torch.testing.assert_close(
            torch.load(result_static / "grid_features.pt"),
            grid_features.reshape(nx, ny, 4)[crop].reshape(-1, 4),
        )
        torch.testing.assert_close(
            torch.load(result_static / "flux_stats.pt"), flux_stats
        )
        for name in stats:
            torch.testing.assert_close(
                torch.load(result_static / f"{name}.pt"), torch.arange(8)
            )
        for split in ("train", "test", "val"):
            for name, (original, expected) in samples.items():
                np.testing.assert_array_equal(
                    np.load(output / "samples" / split / name), expected
                )
                np.testing.assert_array_equal(
                    np.load(source / "samples" / split / name), original
                )


def test_export_without_dataset(tmp_path):
    """Export must work without executing any dataset-dependent cells."""
    pytest.importorskip("marimo", reason="Requires the docs extra")
    docs = tmp_path / "docs"
    (docs / "notebooks").mkdir(parents=True)
    shutil.copyfile(NOTEBOOK, docs / "notebooks" / NOTEBOOK.name)
    exporter = docs / "export_notebooks.py"
    shutil.copyfile(NOTEBOOK.parents[1] / exporter.name, exporter)
    # A second build must overwrite the generated Markdown without prompting.
    for _ in range(2):
        subprocess.run(
            [sys.executable, str(exporter)],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
    exported = (docs / "notebooks" / NOTEBOOK.name).with_suffix(".md")
    markdown = exported.read_text()
    assert "# Creating meps_example_reduced" in markdown
    assert "np.load" in markdown
    assert not (tmp_path / "data").exists()


def test_notebook_edit_link(tmp_path):
    """Generated pages must link to Python sources that exist in Git."""
    pytest.importorskip("sphinx", reason="Requires the docs extra")
    pytest.importorskip("myst_parser", reason="Requires the docs extra")
    pytest.importorskip("sphinx_book_theme", reason="Requires the docs extra")
    docs = tmp_path / "docs"
    (docs / "notebooks").mkdir(parents=True)
    config = (NOTEBOOK.parents[1] / "conf.py").read_text()
    # A small site exercises the real theme without API generation or network
    # inventories; full-site builds are checked separately by the docs jobs.
    (docs / "conf.py").write_text(
        config
        + '\nextensions = ["myst_parser"]\n'
        + "html_logo = None\nhtml_static_path = []\n"
    )
    (docs / "index.md").write_text(
        "# Home\n\n```{toctree}\nnotebooks/example\nguide\n```\n"
    )
    (docs / "guide.md").write_text("# Guide\n")
    (docs / "notebooks" / "example.md").write_text("# Example\n")
    (docs / "notebooks" / "example.py").write_text("# Notebook source\n")
    output = tmp_path / "html"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-b",
            "html",
            str(docs),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    notebook_html = (output / "notebooks" / "example.html").read_text()
    guide_html = (output / "guide.html").read_text()
    edit_root = "https://github.com/mllam/neural-lam/edit/main/docs/"
    assert edit_root + "notebooks/example.py" in notebook_html
    assert edit_root + "notebooks/example.md" not in notebook_html
    assert edit_root + "guide.md" in guide_html
