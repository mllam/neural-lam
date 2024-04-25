"""Modifying a GRIB from Python."""

import earthkit.data
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import pygrib
from neural_lam import constants
from neural_lam.rotate_grid import unrot_lat, unrot_lon, unrotate_latlon


# def plot_data(grb, title, save_path):
#     """Plot the data using Cartopy and save it to a file."""
#     lats, lons = grb.latlons()
#     data = grb.values

#     # Setting up the map projection and the contour plot
#     fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
#     ax.coastlines()
#     contour = ax.contourf(lons, lats, data, levels=np.linspace(data.min(), data.max(), 100), cmap='viridis')
#     ax.set_title(title)
#     plt.colorbar(contour, ax=ax, orientation='vertical')

#     # Save the plot to a file
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close(fig)


def main():
    # Load the grib file 
    original_data = earthkit.data.from_source("file", "/users/clechart/clechart/neural-lam/laf2024042400")
    # subset = original_data.sel(shortName=[x.lower() for x in constants.PARAM_NAMES_SHORT], level=constants.VERTICAL_LEVELS)
    subset = original_data.sel(shortName="u", level=constants.VERTICAL_LEVELS)

    # Load the array to replace the values with 
    replacement_data = np.load("/users/clechart/clechart/neural-lam/wandb/run-20240417_104748-dxnil3vw/files/results/inference/prediction_0.npy")
    cut_values = replacement_data[0,1,:, 26:33].transpose()
    cut_values = np.flip(cut_values, axis=(0,1))
    # cut_values = cut_values.reshape(cut_values.shape[0], -1)
    # cut_values = cut_values.reshape(-1, cut_values.shape[0])

    # Ensure the dimensions match
    assert cut_values.shape == subset.values.shape, "The shapes of the arrays don't match."


    # Find indices where values are around -8
    close_to_minus_eight_subset = np.where((subset.values > -8.5) & (subset.values < -8.0))
    close_to_minus_eight_cut_values = np.where((cut_values > -8.5) & (cut_values < -8.0))

    print(f"Indices in subset.values close to -8: {close_to_minus_eight_subset}")
    print(f"Indices in cut_values close to -8: {close_to_minus_eight_cut_values}")

    # Save the overwritten data 
    md = subset.metadata()
    data_new = earthkit.data.FieldList.from_array(cut_values, md)
    data_new.save("testinho")



def pygrib_plotting():
    # Load original GRIB data 
    original_grib = pygrib.open("/users/clechart/clechart/neural-lam/laf2024042400")
    grb = original_grib.select(shortName = "u", level = 1)[0]
    lats, lons = grb.latlons()
    original_data = grb.values

    # Load transformed GRIB with inference output
    inference_grib = pygrib.open("testinho")
    inf = inference_grib.select(shortName = "u", level = 1)[0]
    predicted_data = inf.values

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
    plt.close(fig)

if __name__ == "__main__":
    main()
    pygrib_plotting()
