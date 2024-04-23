import json
import os

import numpy as np
from earthkit.data import FieldList
import metview as mv

# Need to shuffle the metadata the same way as with constants (see message from Simon)
# Or extract the metadata at the stage where that happens 
# Sort according to the last dimension

class GRIBMetadata:
    def __init__(self, grib_data):
        self.grib_data = grib_data

    def display(self):
        # Display each variable and its metadata in a readable format
        for var, metadata in self.grib_data.items():
            print(f"Variable: {var}")
            for key, value in metadata.items():
                print(f"  {key}: {value}")

    def save_to_file(self, filename):
        # Save the metadata dictionary to a JSON file
        with open(filename, 'w') as f:
            json.dump(self.grib_data, f, indent=4)


def map_zarr_to_grib_metadata(zarr_metadata):
    # Convert metadata from Zarr format to a GRIB-like format
    grib_metadata = {}
    for key, value in zarr_metadata['metadata'].items():
        if '/.zarray' in key:
            var_name = key.split('/')[0]
            array_info = zarr_metadata['metadata'][f'{var_name}/.zarray']
            attrs_info = zarr_metadata['metadata'][f'{var_name}/.zattrs']

            # Rearrange zmetadata according to the shuffling of constants.py
            # -> Makes a subselection of the variables in the zarr archive and shuffles the indices

            if '_ARRAY_DIMENSIONS' in attrs_info:
                dimensions = attrs_info['_ARRAY_DIMENSIONS']
                ny = array_info['shape'][dimensions.index('y_1')] if 'y_1' in dimensions else None
                nx = array_info['shape'][dimensions.index('x_1')] if 'x_1' in dimensions else None

                grib_metadata[var_name] = {
                    'GRIB_paramName': var_name,
                    'GRIB_units': attrs_info.get('units', ''),
                    'GRIB_dataType': array_info['dtype'],
                    'GRIB_totalNumber': array_info['shape'][0] if 'time' in dimensions else 1,
                    'GRIB_gridType': 'regular_ll',
                    'GRIB_Ny': ny,
                    'GRIB_Nx': nx,
                    'GRIB_missingValue': array_info['fill_value']
                }
    return GRIBMetadata(grib_metadata)

def extract_grib_metadata_from_zarr(zarr_path):
    # Load the Zarr dataset's metadata from the .zmetadata JSON file
    metadata_path = os.path.join(zarr_path, '.zmetadata')
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    return map_zarr_to_grib_metadata(metadata)


def complete_data_into_grib(grib_metadata_object):
    data = np.load("/users/clechart/clechart/neural-lam/wandb/run-20240417_104748-dxnil3vw/files/results/inference/prediction_0.npy")
    # How do the pieces of code below work? 
    data = data.set_values(vals)
    mv.write('recip.grib', data)
    ds_new = FieldList.from_array(data, grib_metadata_object)

    print(ds_new)
    return ds_new

if __name__ == "__main__":
    zarr_path = "/users/clechart/clechart/neural-lam/data/cosmo_old/samples/forecast/data_2020011017.zarr"
    grib_metadata_object = extract_grib_metadata_from_zarr(zarr_path)
    grib_metadata_object.display()  # Display the extracted metadata

    # Save metadata to a file
    grib_metadata_object.save_to_file('grib_metadata.json')

    # Reconstruct the GRIB file with the data
    full_set = complete_data_into_grib(grib_metadata_object)


# How does one read an array as a grib? 
# gribfile = xr.open_dataset(joinpath(path,filelist[1]),engine="cfgrib")