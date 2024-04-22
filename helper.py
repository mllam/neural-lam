# Standard library
import os

# Third-party
import xarray as xr

PATH = "data/cosmo/samples/test/"

# Initialize a dictionary to store the top-1 precipitation event for each file
precip_events = {}

for file in os.listdir(PATH):
    print(file)
    ds = xr.open_zarr(os.path.join(PATH, file))

    ds_rechunked = ds.chunk({"time": -1})
    mean_tot_prec = ds_rechunked["TOT_PREC"].mean(dim=["y", "x"]).compute()

    # Find the maximum precipitation value and its corresponding time
    max_precip_value = mean_tot_prec.max().item()
    max_precip_time = mean_tot_prec.where(
        mean_tot_prec == max_precip_value, drop=True
    ).time.values[0]
    MAX_PRECIP_TIME_STR = str(
        max_precip_time
    )  # Convert to string for dictionary key

    # Store the top-1 precipitation event in the dictionary
    precip_events[file] = {
        "max_time": MAX_PRECIP_TIME_STR,
        "max_value": max_precip_value,
    }

# Find the file with the maximum and minimum precipitation values
max_precip_file = max(
    precip_events, key=lambda x: precip_events[x]["max_value"]
)
max_precip_event = precip_events[max_precip_file]

# Sort the precipitation events by maximum value in descending order
sorted_precip_events = sorted(
    precip_events.items(), key=lambda x: x[1]["max_value"], reverse=True
)

# Print the top ten precipitation events
print("Top ten maximum precipitation events:")
for i, (file, event) in enumerate(sorted_precip_events[:10]):
    print(
        f"{i + 1}: {event['max_time']} with a value of "
        f"{event['max_value']} in file {file}"
    )
