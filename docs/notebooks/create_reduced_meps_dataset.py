# Third-party
import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # Standard library
    import argparse
    from pathlib import Path

    # Third-party
    import marimo as mo
    import numpy as np
    import torch

    return Path, argparse, mo, np, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    # Creating meps_example_reduced

    This notebook outlines how the small test dataset
    `meps_example_reduced` was created from `meps_example`. The original
    zipped datasets were 263 MB and 2.6 GB, respectively. See the
    [README](https://github.com/mllam/neural-lam/blob/main/README.md)
    for instructions to download and extract `meps_example`.

    The dataset was reduced by cropping the grid and keeping fewer
    variables. Run this notebook from the repository root with:

    ```bash
    python docs/notebooks/create_reduced_meps_dataset.py
    ```

    The default input is `data/meps_example`; output directories are
    created under `data/meps_example_reduced`. Use `--input-dir` and
    `--output-dir` to select other locations. Rerunning overwrites
    matching output files. The input and output directories must differ.

    The final configuration edit described below remains a manual step.
    Exporting this notebook to Markdown does not run the data processing.
    """
    )
    return


@app.cell
def _(Path, argparse):
    _parser = argparse.ArgumentParser(description="Reduce the MEPS dataset")
    _parser.add_argument(
        "--input-dir", type=Path, default=Path("data/meps_example")
    )
    _parser.add_argument(
        "--output-dir", type=Path, default=Path("data/meps_example_reduced")
    )
    _args = _parser.parse_args()
    input_dir = _args.input_dir.resolve()
    output_dir = _args.output_dir.resolve()
    if not (input_dir / "static" / "nwp_xy.npy").is_file():
        raise FileNotFoundError(
            f"MEPS source dataset not found at {input_dir}. "
            "Download and extract meps_example as described in the README, "
            "or pass --input-dir."
        )
    if output_dir == input_dir:
        raise ValueError("Input and output directories must differ")
    (output_dir / "static").mkdir(parents=True, exist_ok=True)
    for _split in ("train", "test", "val"):
        (output_dir / "samples" / _split).mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reduce the grid

    The number of grid points was reduced to one quarter by halving
    both spatial dimensions. Removing a quarter of the coordinates
    along each outer edge keeps the central grid points centered.
    """
    )
    return


@app.cell
def _(input_dir, np, output_dir):
    grid_xy = np.load(input_dir / "static" / "nwp_xy.npy")
    num_x, num_y = grid_xy.shape[1:]
    x_slice = slice(num_x // 4, 3 * num_x // 4)
    y_slice = slice(num_y // 4, 3 * num_y // 4)
    grid_xy_reduced = grid_xy[:, x_slice, y_slice]
    np.save(output_dir / "static" / "nwp_xy.npy", grid_xy_reduced)
    return grid_xy_reduced, num_x, num_y, x_slice, y_slice


@app.cell
def _(mo):
    mo.md(
        r"""
    Cropping removes the original border. Establish a new perimeter of
    10 grid points, matching the border width in `meps_example`.
    """
    )
    return


@app.cell
def _(grid_xy_reduced, input_dir, np, output_dir):
    _old_border_mask = np.load(input_dir / "static" / "border_mask.npy")
    assert not np.any(_old_border_mask[10:-10, 10:-10])
    assert np.all(_old_border_mask[:10, :])
    assert np.all(_old_border_mask[:, :10])
    assert np.all(_old_border_mask[-10:, :])
    assert np.all(_old_border_mask[:, -10:])

    _border_mask = np.zeros_like(grid_xy_reduced[0], dtype=bool)
    _border_mask[:10] = True
    _border_mask[:, :10] = True
    _border_mask[-10:] = True
    _border_mask[:, -10:] = True
    np.save(output_dir / "static" / "border_mask.npy", _border_mask)
    return


@app.cell
def _(mo):
    mo.md(
        """
    Crop surface geopotential and grid features to the same grid.
    The two-element `flux_stats.pt` vector does not depend on the grid
    shape or the number of variables, so copy it unchanged.
    """
    )
    return


@app.cell
def _(input_dir, np, num_x, num_y, output_dir, torch, x_slice, y_slice):
    _surface_geopotential = np.load(
        input_dir / "static" / "surface_geopotential.npy"
    )
    np.save(
        output_dir / "static" / "surface_geopotential.npy",
        _surface_geopotential[x_slice, y_slice],
    )

    _grid_features = torch.load(input_dir / "static" / "grid_features.pt")
    print(_grid_features.shape)
    # Reshape to the grid, crop, then flatten the spatial dimensions again.
    _grid_features_reduced = _grid_features.reshape(num_x, num_y, 4)[
        x_slice, y_slice, :
    ].reshape((-1, 4))
    torch.save(
        _grid_features_reduced, output_dir / "static" / "grid_features.pt"
    )
    torch.save(
        torch.load(input_dir / "static" / "flux_stats.pt"),
        output_dir / "static" / "flux_stats.pt",
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Reduce the variables

    Keep the first eight variables, including their weights and
    normalization statistics.
    """
    )
    return


@app.cell
def _(input_dir, np, output_dir, torch):
    num_vars = 8
    _parameter_weights = np.load(input_dir / "static" / "parameter_weights.npy")
    np.save(
        output_dir / "static" / "parameter_weights.npy",
        _parameter_weights[:num_vars],
    )
    for _name in ("diff_mean", "diff_std", "parameter_mean", "parameter_std"):
        _old_file = torch.load(input_dir / "static" / f"{_name}.pt")
        torch.save(_old_file[:num_vars], output_dir / "static" / f"{_name}.pt")
    return (num_vars,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reduce the samples

    Each of `samples/train`, `samples/test`, and `samples/val` contains:

    - `nwp_YYYYMMDDHH_mbrXXX.npy`: forecast time, two spatial dimensions,
      then variables. Crop the spatial dimensions and keep eight variables.
    - `wtr_YYYYMMDDHH.npy`: two spatial dimensions, both cropped.
    - `nwp_toa_downwelling_shortwave_flux_YYYYMMDDHH.npy`: forecast time
      followed by two spatial dimensions. Keep every forecast time.

    Here `YYYYMMDDHH` is a date and hour, and `XXX` is a three-digit
    member index. In the original example, the state array had shape
    `(65, 268, 238, 18)` and the shortwave-flux array had shape
    `(65, 268, 238)`.
    """
    )
    return


@app.cell
def _(input_dir, np, num_vars, output_dir, x_slice, y_slice):
    for _sample in ("train", "test", "val"):
        for _file in sorted((input_dir / "samples" / _sample).iterdir()):
            _data = np.load(_file)
            if "mbr" in _file.name:
                _data = _data[:, x_slice, y_slice, :num_vars]
            elif "wtr" in _file.name:
                _data = _data[x_slice, y_slice]
            else:
                _data = _data[:, x_slice, y_slice]
            np.save(output_dir / "samples" / _sample / _file.name, _data)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Update the configuration manually

    Finally, copy `data_config.yaml` from the source dataset into the
    output directory. Truncate the variable units, long names and short
    names to the first eight entries, and set the new grid shape.

    In the original reduction, unit descriptions containing `^` were
    changed to `**` to avoid requiring LaTeX in GitHub CI. The resulting
    `data/meps_example_reduced` directory was then archived and uploaded
    to a European Weather Cloud S3 bucket.
    """
    )
    return


if __name__ == "__main__":
    app.run()
