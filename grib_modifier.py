# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cf
import earthkit.data
import numpy as np
import pygrib
from matplotlib import pyplot as plt

# First-party
from neural_lam import constants


def plot_data(
    grb, title, ax, projection, vmin, vmax, color_map="plasma", num_contours=100
):
    """Plot the data using Cartopy."""
    lats, lons = grb.latlons()
    data = grb.values

    ax.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
    ax.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
    ax.set_title(title)

    contour = ax.contourf(
        lons,
        lats,
        data,
        transform=projection(),
        levels=np.linspace(vmin, vmax, num_contours),
        cmap=color_map,
    )
    return contour


def modify_data():
    """Fit the numpy values into GRIB format."""
    # Load the original grib file
    original_data = earthkit.data.from_source(
        "file", "/users/clechart/clechart/neural-lam/laf2024042400"
    )
    subset = original_data.sel(shortName="u", level=constants.VERTICAL_LEVELS)

    # Load the array to replace the values with
    replacement_data = np.load(
        "/users/clechart/clechart/neural-lam/"
        "wandb/run-20240417_104748-dxnil3vw/files/"
        "results/inference/prediction_0.npy"
    )
    original_cut = replacement_data[0, 1, :, 26:33].reshape(582, 390, 7)
    cut_values = np.moveaxis(original_cut, [-3, -2, -1], [-1, -2, -3])

    # Create the modified GRIB file with the predicted data
    modified_grib_path = "/users/clechart/clechart/neural-lam/modified_grib"
    md = subset.metadata()
    data_new = earthkit.data.FieldList.from_array(cut_values, md)
    data_new.save(modified_grib_path)


def generate_plot():
    """Plot the original GRIB entries against transformed inference."""
    # Open the original and modified GRIB files
    original_grib = pygrib.open(
        "/users/clechart/clechart/neural-lam/laf2024042400"
    )
    grb_original = original_grib.select(
        shortName="u", level=constants.VERTICAL_LEVELS[0]
    )[0]

    predicted_grib = pygrib.open(
        "/users/clechart/clechart/neural-lam/modified_grib"
    )
    grb_predicted = predicted_grib.select(
        shortName="u", level=constants.VERTICAL_LEVELS[0]
    )[0]

    # Determine the global min and max values for the colorbar
    vmin = min(grb_original.values.min(), grb_predicted.values.min())
    vmax = max(grb_original.values.max(), grb_predicted.values.max())

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    contour1 = plot_data(
        grb_original, "Original Data", ax1, ccrs.PlateCarree, vmin, vmax
    )
    _ = plot_data(
        grb_predicted, "Predicted Data", ax2, ccrs.PlateCarree, vmin, vmax
    )

    plt.subplots_adjust(hspace=0.1, wspace=0.05)
    colorbar_ax = fig.add_axes(
        [0.15, 0.08, 0.7, 0.02]
    )  # Position for the colorbar
    fig.colorbar(
        contour1, cax=colorbar_ax, orientation="horizontal", shrink=0.5
    )

    plt.savefig("completed_grib.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    modify_data()
    generate_plot()
