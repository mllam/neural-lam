# Third-party
import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    # Third-party
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Creating meps_example_reduced
    This notebook outlines how the small-size test dataset ```meps_example_reduced``` was created based on the slightly larger dataset ```meps_example```. The zipped up datasets are 263 MB and 2.6 GB, respectively. See [README.md](../../README.md) for info on how to download ```meps_example```.

    The dataset was reduced in size by reducing the number of grid points and variables.
    """
    )
    return


@app.cell
def _():
    # Cell tags: skip-execution
    # Standard library
    import os

    # Third-party
    import numpy as np
    import torch

    return np, os, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The number of grid points was reduced to 1/4 by halving the number of coordinates in both the x and y direction. This was done by removing a quarter of the grid points along each outer edge, so the center grid points would stay centered in the new set.
    """
    )
    return


@app.cell
def _(np):
    # Cell tags: skip-execution
    # Load existing grid
    grid_xy = np.load("data/meps_example/static/nwp_xy.npy")
    # Get slices in each dimension by cutting off a quarter along each edge
    num_x, num_y = grid_xy.shape[1:]
    x_slice = slice(num_x // 4, 3 * num_x // 4)
    y_slice = slice(num_y // 4, 3 * num_y // 4)
    # Index and save reduced grid
    grid_xy_reduced = grid_xy[:, x_slice, y_slice]
    np.save("data/meps_example_reduced/static/nwp_xy.npy", grid_xy_reduced)
    return grid_xy_reduced, num_x, num_y, x_slice, y_slice


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This cut out the border, so a new perimeter of 10 grid points was established as border (10 was also the border size in the original "meps_example").
    """
    )
    return


@app.cell
def _(grid_xy_reduced, np):
    # Cell tags: skip-execution
    # Outer 10 grid points are border
    old_border_mask = np.load("data/meps_example/static/border_mask.npy")
    assert np.all(old_border_mask[10:-10, 10:-10] == False)
    assert np.all(old_border_mask[:10, :] == True)
    assert np.all(old_border_mask[:, :10] == True)
    assert np.all(old_border_mask[-10:, :] == True)
    assert np.all(old_border_mask[:, -10:] == True)

    # Create new array with False everywhere but the outer 10 grid points
    border_mask = np.zeros_like(grid_xy_reduced[0, :, :], dtype=bool)
    border_mask[:10] = True
    border_mask[:, :10] = True
    border_mask[-10:] = True
    border_mask[:, -10:] = True
    np.save("data/meps_example_reduced/static/border_mask.npy", border_mask)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    A few other files also needed to be copied using only the new reduced grid
    """
    )
    return


@app.cell
def _(np, num_x, num_y, torch, x_slice, y_slice):
    # Cell tags: skip-execution
    # Load surface_geopotential.npy, index only values from the reduced grid, and save to new file
    surface_geopotential = np.load(
        "data/meps_example/static/surface_geopotential.npy"
    )
    surface_geopotential_reduced = surface_geopotential[x_slice, y_slice]
    np.save(
        "data/meps_example_reduced/static/surface_geopotential.npy",
        surface_geopotential_reduced,
    )

    # Load pytorch file grid_features.pt
    grid_features = torch.load("data/meps_example/static/grid_features.pt")
    # Index only values from the reduced grid.
    # First reshape from (num_grid_points_total, 4) to (num_grid_points_x, num_grid_points_y, 4),
    # then index, then reshape back to new total number of grid points
    print(grid_features.shape)
    grid_features_new = grid_features.reshape(num_x, num_y, 4)[
        x_slice, y_slice, :
    ].reshape((-1, 4))
    # Save to new file
    torch.save(
        grid_features_new, "data/meps_example_reduced/static/grid_features.pt"
    )

    # flux_stats.pt is just a vector of length 2, so the grid shape and variable changes does not change this file
    torch.save(
        torch.load("data/meps_example/static/flux_stats.pt"),
        "data/meps_example_reduced/static/flux_stats.pt",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The number of variables was reduced by truncating the variable list to the first 8.
    """
    )
    return


@app.cell
def _(np, torch):
    # Cell tags: skip-execution
    num_vars = 8

    # Load parameter_weights.npy, truncate to first 8 variables, and save to new file
    parameter_weights = np.load(
        "data/meps_example/static/parameter_weights.npy"
    )
    parameter_weights_reduced = parameter_weights[:num_vars]
    np.save(
        "data/meps_example_reduced/static/parameter_weights.npy",
        parameter_weights_reduced,
    )

    # Do the same for following 4 pytorch files
    for file in ["diff_mean", "diff_std", "parameter_mean", "parameter_std"]:
        old_file = torch.load(f"data/meps_example/static/{file}.pt")
        new_file = old_file[:num_vars]
        torch.save(new_file, f"data/meps_example_reduced/static/{file}.pt")
    return (num_vars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Lastly the files in each of the directories train, test, and val have to be reduced. The folders all have the same structure with files of the following types:
    ```
    nwp_YYYYMMDDHH_mbrXXX.npy
    wtr_YYYYMMDDHH.npy
    nwp_toa_downwelling_shortwave_flux_YYYYMMDDHH.npy
    ```
    with ```YYYYMMDDHH``` being some date with hours, and ```XXX``` being some 3-digit integer.

    The first type of file has x and y in dimensions 1 and 2, and variable index in dimension 3. Dimension 0 is unchanged.
    The second type has has x and y in dimensions 1 and 2. Dimension 0 is unchanged.
    The last type has just x and y as the only 2 dimensions.
    """
    )
    return


@app.cell
def _(np):
    # Cell tags: skip-execution
    print(
        np.load(
            "data/meps_example/samples/train/nwp_2022040100_mbr000.npy"
        ).shape
    )
    print(
        np.load(
            "data/meps_example/samples/train/nwp_toa_downwelling_shortwave_flux_2022040112.npy"
        ).shape
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The following loop goes through each file in each sample folder and indexes them according to the dimensions given by the file name.
    """
    )
    return


@app.cell
def _(np, num_vars, os, x_slice, y_slice):
    # Cell tags: skip-execution
    for sample in ["train", "test", "val"]:
        files = os.listdir(f"data/meps_example/samples/{sample}")

        for f in files:
            data = np.load(f"data/meps_example/samples/{sample}/{f}")
            if "mbr" in f:
                data = data[:, x_slice, y_slice, :num_vars]
            elif "wtr" in f:
                data = data[x_slice, y_slice]
            else:
                data = data[:, x_slice, y_slice]
            np.save(f"data/meps_example_reduced/samples/{sample}/{f}", data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Lastly, the file ```data_config.yaml``` is modified manually by truncating the variable units, long and short names, and setting the new grid shape. Also the unit descriptions containing ```^``` was automatically parsed using latex, and to avoid having to install latex in the GitHub CI/CD pipeline, this was changed to ```**```.

    This new config file was placed in ```data/meps_example_reduced```, and that directory was then zipped and placed in a European Weather Cloud S3 bucket.
    """
    )
    return


if __name__ == "__main__":
    app.run()
