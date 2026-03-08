"""
Test script for neural_lam/vis.py functions.
Tests all 3 functions with real meps_example data.
Run from repo root: python test_vis.py
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, saves to file
import matplotlib.pyplot as plt

from neural_lam.datastore.npyfilesmeps.store import NpyFilesDatastoreMEPS
import neural_lam.vis as vis

import os

# ── Load datastore ───────────────────────────────────────────
print("Loading datastore...")

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "meps_example", "meps.datastore.yaml"
)
print(f"Config path: {config_path}")
print(f"File exists: {os.path.exists(config_path)}")

datastore = NpyFilesDatastoreMEPS(config_path=config_path)

var_names = datastore.get_vars_names("state")
var_units = datastore.get_vars_units("state")
n_vars = len(var_names)
n_grid = datastore.num_grid_points

print(f"Variables ({n_vars}): {var_names}")
print(f"Grid points: {n_grid}")
print(f"Grid shape: {datastore.grid_shape_state}")

# ════════════════════════════════════════════════════════════
# TEST 1: plot_error_map — normal case
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TEST 1: plot_error_map — normal case")
print("="*50)

pred_steps = 4
# Shape must be (pred_steps, n_vars) as function expects
errors = torch.rand(pred_steps, n_vars) * 100
print(f"errors shape: {errors.shape}")
print(f"Expected: (pred_steps={pred_steps}, n_vars={n_vars})")

try:
    fig = vis.plot_error_map(errors=errors, datastore=datastore,
                              title="Test Error Map")
    fig.savefig("output_test1_error_map.png", 
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ TEST 1 PASSED — saved output_test1_error_map.png")
except Exception as e:
    print(f"❌ TEST 1 FAILED: {e}")

# ════════════════════════════════════════════════════════════
# TEST 2: plot_error_map — BUG TEST: zero errors
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TEST 2: plot_error_map — zero error bug")
print("="*50)

# Set one variable to all zeros → should cause div by zero
errors_zero = torch.rand(pred_steps, n_vars) * 100
errors_zero[:, 0] = 0.0  # first variable has zero error
print(f"errors_zero shape: {errors_zero.shape}")
print(f"Variable '{var_names[0]}' set to all zeros")

try:
    fig = vis.plot_error_map(errors=errors_zero, datastore=datastore,
                              title="Zero Error Bug Test")
    fig.savefig("output_test2_zero_error.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ TEST 2 PASSED (no crash)")
    # Check if NaN appeared
    import matplotlib.cm as cm
    print("  Check output_test2_zero_error.png for NaN/blank column")
except Exception as e:
    print(f"❌ TEST 2 FAILED (div by zero bug confirmed): {e}")

# ════════════════════════════════════════════════════════════
# TEST 3: plot_error_map — BUG TEST: 1D errors
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TEST 3: plot_error_map — 1D tensor bug")
print("="*50)

errors_1d = torch.rand(n_vars) * 100
print(f"errors_1d shape: {errors_1d.shape} (1D — should crash without fix)")

try:
    fig = vis.plot_error_map(errors=errors_1d, datastore=datastore,
                              title="1D Error Bug Test")
    fig.savefig("output_test3_1d_error.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ TEST 3 PASSED (fix working)")
except Exception as e:
    print(f"❌ TEST 3 FAILED (original bug still present): {e}")

# ════════════════════════════════════════════════════════════
# TEST 4: plot_spatial_error — BUG TEST: grid_shape_state.x
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TEST 4: plot_spatial_error — grid_shape_state attribute bug")
print("="*50)

# Check what grid_shape_state actually is
print(f"grid_shape_state = {datastore.grid_shape_state}")
print(f"type = {type(datastore.grid_shape_state)}")

# Try accessing .x and .y like the code does
try:
    x = datastore.grid_shape_state.x
    y = datastore.grid_shape_state.y
    print(f"  .x = {x}, .y = {y}")
    print("  grid_shape_state has .x and .y attributes")
except AttributeError as e:
    print(f"❌ BUG CONFIRMED: {e}")
    print("  grid_shape_state is a tuple, not an object with .x/.y")
    print("  plot_spatial_error will crash when called")

# Try calling the function anyway
error = torch.rand(n_grid)
print(f"\nerror shape: {error.shape}")

try:
    fig = vis.plot_spatial_error(
        error=error,
        datastore=datastore,
        title="Spatial Error Test"
    )
    fig.savefig("output_test4_spatial_error.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ TEST 4 PASSED")
except Exception as e:
    print(f"❌ TEST 4 FAILED: {e}")

# ════════════════════════════════════════════════════════════
# TEST 5: plot_prediction
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("TEST 5: plot_prediction")
print("="*50)

import xarray as xr
import traceback

try:
    da_state = datastore.get_dataarray("state", "train")
    print(f"State data shape: {da_state.shape}")
    print(f"State data dims: {da_state.dims}")

    da_pred = da_state.isel(
        analysis_time=0,
        elapsed_forecast_duration=0,
        ensemble_member=0,
        state_feature=0
    )
    da_target = da_state.isel(
        analysis_time=0,
        elapsed_forecast_duration=1,
        ensemble_member=0,
        state_feature=0
    )

    print(f"da_pred shape: {da_pred.shape}")
    print(f"da_pred dims: {da_pred.dims}")
    print(f"da_pred coords: {dict(da_pred.coords)}")

    # Debug what plot_prediction needs
    print(f"\nextent = {datastore.get_xy_extent('state')}")
    print(f"boundary_mask shape: {datastore.boundary_mask.shape}")
    print(f"coords_projection: {datastore.coords_projection}")

    # Try unstacking grid coords
    da_unstacked = datastore.unstack_grid_coords(da_pred)
    print(f"\nAfter unstack shape: {da_unstacked.shape}")
    print(f"After unstack dims: {da_unstacked.dims}")

    fig = vis.plot_prediction(
        datastore=datastore,
        da_prediction=da_pred,
        da_target=da_target,
        title=f"Test Prediction: {var_names[0]}"
    )
    fig.savefig("output_test5_prediction.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("✅ TEST 5 PASSED")
except Exception as e:
    print(f"❌ TEST 5 FAILED: {e}")
    traceback.print_exc()
# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("SUMMARY — check these output files:")
print("="*50)
print("output_test1_error_map.png   — normal error heatmap")
print("output_test2_zero_error.png  — zero error div by zero bug")
print("output_test3_1d_error.png    — 1D tensor bug")
print("output_test4_spatial_error.png — grid_shape_state bug")
print("output_test5_prediction.png  — prediction vs ground truth")