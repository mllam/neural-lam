"""Modifying a GRIB from Python."""

import earthkit.data
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import pygrib
from neural_lam import constants


def plot_data(grb, title, save_path):
    """Plot the data using Cartopy and save it to a file."""
    lats, lons = grb.latlons()
    data = grb.values

    # Setting up the map projection and the contour plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.coastlines()
    contour = ax.contourf(lons, lats, data, levels=np.linspace(data.min(), data.max(), 100), cmap='viridis')
    ax.set_title(title)
    plt.colorbar(contour, ax=ax, orientation='vertical')

    # Save the plot to a file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


def main():
    # Load the grib file 
    ds = earthkit.data.from_source("file", "/users/clechart/clechart/neural-lam/laf2024042400")
    # subset = ds.sel(shortName=[x.lower() for x in constants.PARAM_NAMES_SHORT], level=constants.VERTICAL_LEVELS)
    subset = ds.sel(shortName="u", level=constants.VERTICAL_LEVELS)

    # Load the array to replace the values with 
    replacement = np.load("/users/clechart/clechart/neural-lam/wandb/run-20240417_104748-dxnil3vw/files/results/inference/prediction_0.npy")
    cut_values = replacement[0,1,:, 26:33].transpose()
    
    md = subset.metadata()
    ds_new = earthkit.data.FieldList.from_array(cut_values, md)
    ds_new.save("testinho")



def pygrib_plotting():
    # FIXME fix those damn names
    grbs = pygrib.open("/users/clechart/clechart/neural-lam/laf2024042400")
    first_grb = grbs.select(shortName = "u", level = 1)[0] # Select the first one
    # plot_data(first_grb, "Original grib file", "test.png")

    grps = pygrib.open("testinho")
    first_grps = grps.select(shortName = "u", level = 1)[0]
    second_grps = first_grps.values
    # plot_data(second_grps, "Extracted values from the inference", "testinho.png")


    """Plot the data using Cartopy and save it to a file."""
    lats, lons = first_grb.latlons()
    original_data = first_grb.values
    predicted_data = second_grps
    vmin = original_data.min()
    vmax = original_data.max()

    # Setting up the map projection and the contour plot
    fig, axes = plt.subplots(
        2,
        1,
        figsize=constants.FIG_SIZE,
        subplot_kw={"projection": constants.SELECTED_PROJ},
    )
    for axes, data in zip(axes, (original_data, predicted_data)):
        contour_set = axes.contourf(
            lons,
            lats,
            data,
            transform=constants.SELECTED_PROJ,
            cmap="plasma",
            levels=np.linspace(vmin, vmax, num=100),
        )
        axes.add_feature(cf.BORDERS, linestyle="-", edgecolor="black")
        axes.add_feature(cf.COASTLINE, linestyle="-", edgecolor="black")
        axes.gridlines(
            crs=constants.SELECTED_PROJ,
            draw_labels=False,
            linewidth=0.5,
            alpha=0.5,
        )

    # Ticks and labels
    # axes[0].set_title("Ground Truth", size=15)
    # axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(contour_set, orientation="horizontal", aspect=20)
    cbar.ax.tick_params(labelsize=10)

    # Save the plot to a file
    plt.savefig("megamind_metadata.png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


if __name__ == "__main__":
    pygrib_plotting()
    # main()
