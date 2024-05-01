# Third-party
from datetime import datetime, timedelta
import earthkit.data
import numpy as np

# First-party
from neural_lam import constants


def modify_data(prediction: np.array):
    """Fit the numpy values into GRIB format."""

    indices = precompute_variable_indices()
    time_steps = generate_time_steps()
    vertical_levels = [1, 5, 13, 22, 38, 41, 60]

    # Initialize final data object 
    final_data = earthkit.data.FieldList()

    # Loop through all the time steps and all the variables
    for time_idx, date_str in time_steps.items():

        # ATN between 3D and 2D - 7vs1 lvl (for reshaping) 
        for variable in constants.PARAM_NAMES_SHORT:
            # here find the key of the cariable in constants.is_3D and if == 7, assign a cut of 7 on the 
            # reshape. Else 1 
            shape_val = 7 if constants.IS_3D[variable] else 1
            # Find the value range to sample 
            value_range = indices[variable]

            sample_file = constants.SAMPLE_GRIB
            if variable == "RELHUM":
                variable = "r"
                sample_file = constants.SAMPLE_Z_GRIB

            # Load the sample grib file
            original_data = earthkit.data.from_source(
                "file", sample_file
            )
        
            subset = original_data.sel(shortName= variable.lower(), level=vertical_levels)
            md = subset.metadata()

            # Cut the datestring into date and time and then override all 
            # values in md 
            date = date_str[:8]
            time = date_str[8:]

            # Assuming md is a list of metadata dictionaries
            for metadata in md:
                metadata.override({
                    "date": date,
                    "time": time
                })
            if len(md)>0:
                # Load the array to replace the values with
                # We need to still save it as a .npy object and pass it on as an argument to this function 
                replacement_data = np.load(prediction)        
                original_cut = replacement_data[0, time_idx, :, min(value_range):max(value_range)+1].reshape(582, 390, shape_val)
                cut_values = np.moveaxis(original_cut, [-3, -2, -1], [-1, -2, -3])
                # Can we stack Fieldlists? 
                data_new = earthkit.data.FieldList.from_array(cut_values, md)
                final_data += data_new

        # Create the modified GRIB file with the predicted data
        modified_grib_path =f"lightning_logs/prediction_{date_str}"
        final_data.save(modified_grib_path)

# This function is taken from ar_model, need to just use self when I go 
# put this function into on_predict_epoch_end()
def precompute_variable_indices():
    """
    Precompute indices for each variable in the input tensor
    """
    variable_indices = {}
    all_vars = []
    index = 0
    # Create a list of tuples for all variables, using level 0 for 2D
    # variables
    for var_name in constants.PARAM_NAMES_SHORT:
        if constants.IS_3D[var_name]:
            for level in constants.VERTICAL_LEVELS:
                all_vars.append((var_name, level))
        else:
            all_vars.append((var_name, 0))  # Use level 0 for 2D variables

    # Sort the variables based on the tuples
    sorted_vars = sorted(all_vars)

    for var in sorted_vars:
        var_name, level = var
        if var_name not in variable_indices:
            variable_indices[var_name] = []
        variable_indices[var_name].append(index)
        index += 1

    return variable_indices


def generate_time_steps():
    # Parse the times
    base_time = constants.EVAL_DATETIMES[0]
    if isinstance(base_time, str):
        base_time = datetime.strptime(base_time, "%Y%m%d%H")
    else:
        base_time = base_time 
    time_steps = {}
    # Generate dates for each step
    for i in range(constants.EVAL_HORIZON - 2):
        # Compute the new date by adding the step interval in hours - 3
        new_date = base_time + timedelta(hours=i * constants.TRAIN_HORIZON)
        # Format the date back
        time_steps[i] = new_date.strftime("%Y%m%d%H")
    
    return time_steps



if __name__ == "__main__":
    precompute_variable_indices()
    time_steps = generate_time_steps()
    modify_data(prediction="/users/clechart/neural-lam/templates/predictions.npy")
