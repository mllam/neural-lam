import os

import xarray as xr

path = "data/cosmo/samples/train/"

# Initialize a dictionary to store the top-1 precipitation event for each file
precip_events = {}

for file in os.listdir(path):
    print(file)
    ds = xr.open_zarr(os.path.join(path, file))

    ds_rechunked = ds.chunk({'time': -1})
    mean_tot_prec = ds_rechunked['TOT_PREC'].max(dim=['y_1', 'x_1']).compute()

    # Find the maximum precipitation value and its corresponding time
    max_precip_value = mean_tot_prec.max().item()
    max_precip_time = mean_tot_prec.where(
        mean_tot_prec == max_precip_value,
        drop=True).time.values[0]
    max_precip_time_str = str(max_precip_time)  # Convert to string for dictionary key

    # Find the minimum precipitation value and its corresponding time
    min_precip_value = mean_tot_prec.min().item()
    min_precip_time = mean_tot_prec.where(
        mean_tot_prec == min_precip_value,
        drop=True).time.values[0]
    min_precip_time_str = str(min_precip_time)  # Convert to string for dictionary key

    # Store the top-1 precipitation event in the dictionary
    precip_events[file] = {
        'max_time': max_precip_time_str,
        'max_value': max_precip_value,
        'min_time': min_precip_time_str,
        'min_value': min_precip_value
    }

# Find the file with the maximum and minimum precipitation values
max_precip_file = max(precip_events, key=lambda x: precip_events[x]['max_value'])
min_precip_file = min(precip_events, key=lambda x: precip_events[x]['min_value'])
max_precip_event = precip_events[max_precip_file]
min_precip_event = precip_events[min_precip_file]

print(f"Maximum precipitation event: {max_precip_event} in file {max_precip_file}")
print(f"Minimum precipitation event: {min_precip_event} in file {min_precip_file}")
