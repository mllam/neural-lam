"""Check the dataset tutorial with small, deterministic input arrays."""

# Standard library
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
    nx, ny, nvars = 50, 54, 10
    crop = (slice(12, 37), slice(13, 40))
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
        )
        result_static = output / "static"
        np.testing.assert_array_equal(
            np.load(result_static / "nwp_xy.npy"), grid[:, crop[0], crop[1]]
        )
        np.testing.assert_array_equal(
            np.load(result_static / "surface_geopotential.npy"), surface[crop]
        )
        expected_border = np.ones((25, 27), dtype=bool)
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


def test_generated_page_edit_links(tmp_path):
    """Generated pages must link to their tracked Python sources."""
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
        "# Home\n\n```{toctree}\nnotebooks/example\n"
        "api/neural_lam/example/index\nguide\n```\n"
    )
    (docs / "guide.md").write_text("# Guide\n")
    (docs / "notebooks" / "example.md").write_text("# Example\n")
    (docs / "notebooks" / "example.py").write_text("# Notebook source\n")
    (docs / "api" / "neural_lam" / "example").mkdir(parents=True)
    (docs / "api" / "neural_lam" / "example" / "index.md").write_text("# API\n")
    (tmp_path / "neural_lam").mkdir()
    (tmp_path / "neural_lam" / "example.py").write_text("# Module source\n")
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
    )
    notebook_html = (output / "notebooks" / "example.html").read_text()
    api_html = (
        output / "api" / "neural_lam" / "example" / "index.html"
    ).read_text()
    guide_html = (output / "guide.html").read_text()
    edit_root = "https://github.com/mllam/neural-lam/edit/main/"
    assert edit_root + "docs/notebooks/example.py" in notebook_html
    assert edit_root + "docs/notebooks/example.md" not in notebook_html
    assert edit_root + "neural_lam/example.py" in api_html
    assert edit_root + "docs/guide.md" in guide_html
