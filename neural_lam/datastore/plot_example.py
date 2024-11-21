# Third-party
import matplotlib.pyplot as plt

# Local
from . import DATASTORES, init_datastore


def plot_example_from_datastore(
    category,
    datastore,
    col_dim,
    split="train",
    standardize=True,
    selection={},
    index_selection={},
):
    """
    Create a plot of the data from the datastore.

    Parameters
    ----------
    category : str
        Category of data to plot, one of "state", "forcing", or "static".
    datastore : Datastore
        Datastore to retrieve data from.
    col_dim : str
        Dimension to use for plot facetting into columns. This can be a
        template string that can be formatted with the category name.
    split : str, optional
        Split of data to plot, by default "train".
    standardize : bool, optional
        Whether to standardize the data before plotting, by default True.
    selection : dict, optional
        Selections to apply to the dataarray, for example
        `time="1990-09-03T0:00" would select this single timestep, by default
        {}.
    index_selection: dict, optional
        Index-based selection to apply to the dataarray, for example
        `time=0` would select the first item along the `time` dimension, by
        default {}.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    da = datastore.get_dataarray(category=category, split=split)
    if standardize:
        da_stats = datastore.get_standardization_dataarray(category=category)
        da = (da - da_stats[f"{category}_mean"]) / da_stats[f"{category}_std"]
    da = datastore.unstack_grid_coords(da)

    if len(selection) > 0:
        da = da.sel(**selection)
    if len(index_selection) > 0:
        da = da.isel(**index_selection)

    col = col_dim.format(category=category)

    # check that the column dimension exists and that the resulting shape is 2D
    if col not in da.dims:
        raise ValueError(f"Column dimension {col} not found in dataarray.")
    da_col_item = da.isel({col: 0}).squeeze()
    if not len(da_col_item.shape) == 2:
        raise ValueError(
            f"Column dimension {col} and selection {selection} does not "
            "result in a 2D dataarray. Please adjust the column dimension "
            "and/or selection. Instead the resulting dataarray is:\n"
            f"{da_col_item}"
        )

    crs = datastore.coords_projection
    col_wrap = min(4, int(da[col].count()))
    g = da.plot(
        x="x",
        y="y",
        col=col,
        col_wrap=col_wrap,
        subplot_kws={"projection": crs},
        transform=crs,
        size=4,
    )
    for ax in g.axes.flat:
        ax.coastlines()
        ax.gridlines(draw_labels=["left", "bottom"])
        ax.set_extent(datastore.get_xy_extent(category=category), crs=crs)

    return g.fig


if __name__ == "__main__":
    # Standard library
    import argparse

    def _parse_dict(arg_str):
        key, value = arg_str.split("=")
        for op in [int, float]:
            try:
                value = op(value)
                break
            except ValueError:
                pass
        return key, value

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--datastore_kind",
        type=str,
        choices=DATASTORES.keys(),
        default="mdp",
        help="Kind of datastore to use",
    )
    parser.add_argument(
        "--datastore_config_path",
        type=str,
        default=None,
        help="Path for the datastore config",
    )
    parser.add_argument(
        "--category",
        default="state",
        help="Category of data to plot",
        choices=["state", "forcing", "static"],
    )
    parser.add_argument(
        "--split", default="train", help="Split of data to plot"
    )
    parser.add_argument(
        "--col-dim",
        default="{category}_feature",
        help="Dimension to use for plot facetting into columns",
    )
    parser.add_argument(
        "--disable-standardize",
        dest="standardize",
        action="store_false",
        help="Disable standardization of data",
    )
    # add the ability to create dictionary of kwargs
    parser.add_argument(
        "--selection",
        nargs="+",
        default=[],
        type=_parse_dict,
        help="Selections to apply to the dataarray, for example "
        "`time='1990-09-03T0:00' would select this single timestep",
    )
    parser.add_argument(
        "--index-selection",
        nargs="+",
        default=[],
        type=_parse_dict,
        help="Index-based selection to apply to the dataarray, for example "
        "`time=0` would select the first item along the `time` dimension",
    )
    args = parser.parse_args()

    assert (
        args.datastore_config_path is not None
    ), "Specify your datastore config with --datastore_config_path"

    selection = dict(args.selection)
    index_selection = dict(args.index_selection)

    # check that column dimension is not in the selection
    if args.col_dim.format(category=args.category) in selection:
        raise ValueError(
            f"Column dimension {args.col_dim.format(category=args.category)} "
            f"cannot be in the selection ({selection}). Please adjust the "
            "column dimension and/or selection."
        )

    datastore = init_datastore(
        datastore_kind=args.datastore_kind,
        config_path=args.datastore_config_path,
    )

    plot_example_from_datastore(
        args.category,
        datastore,
        split=args.split,
        col_dim=args.col_dim,
        standardize=args.standardize,
        selection=selection,
        index_selection=index_selection,
    )
    plt.show()
