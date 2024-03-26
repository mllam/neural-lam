# Standard library
import argparse
import os

# Third-party
import xarray as xr
from dask.diagnostics import ProgressBar
from numcodecs import Blosc


def process_grib_files(paths, selected_vars, indexpath):
    """Process grib2 files and return a dataset."""
    datasets = []
    for filters, variables in selected_vars.items():
        backend_kwargs = {
            "errors": "ignore",
            "encode_cf": ["time", "geography"],
            "indexpath": indexpath + "{path}.{short_hash}.idx",
            "filter_by_keys": {},
        }
        print(f"Processing {filters} with variables {variables}")
        if filters == "shortName":
            for var in variables:
                backend_kwargs["filter_by_keys"] = {"shortName": var}
                with ProgressBar():
                    ds = xr.open_mfdataset(
                        paths,
                        engine="cfgrib",
                        backend_kwargs=backend_kwargs,
                        combine="nested",
                        concat_dim="time",
                        parallel=True,
                        chunks={"time": 1},
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                    )
                datasets.append(ds)
        else:
            backend_kwargs["filter_by_keys"] = {"typeOfLevel": filter}
            if filter == "surface":
                backend_kwargs["filter_by_keys"]["stepType"] = "instant"
            with ProgressBar():
                ds = xr.open_mfdataset(
                    paths,
                    engine="cfgrib",
                    backend_kwargs=backend_kwargs,
                    combine="nested",
                    concat_dim="time",
                    parallel=True,
                    chunks={"time": 1},
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                )
            datasets.append(ds[vars])
    return xr.merge(datasets, compat="minimal")


def main(
    data_in, data_train, data_test, indexpath, selected_vars, selected_vars_2
):
    """Process grib2 files and save them to zarr format."""
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    all_files = [
        os.path.join(root, f)
        for root, dirs, files in os.walk(data_in)
        for f in files
        if root.endswith("det") and f.startswith("laf")
    ]

    def years_filter(y):
        return [f for f in all_files if y(int(f[-10:-6]))]

    def process(y, variables):
        return process_grib_files(years_filter(y), variables, indexpath)

    print("Processing training data")
    ds_train = xr.merge(
        [
            process(lambda x: 2015 <= x <= 2019, selected_vars),
            process(lambda x: 2015 <= x <= 2019, selected_vars_2),
        ],
        compat="override",
    ).rename_vars({"pp": "PP"})

    print("Processing testing data")
    ds_test = xr.merge(
        [
            process(lambda x: x == 2020, selected_vars),
            process(lambda x: x == 2020, selected_vars_2),
        ],
        compat="override",
    ).rename_vars({"pp": "PP"})

    for ds, path in zip([ds_train, ds_test], [data_train, data_test]):
        print(f"Saving Zarr to {path}")
        with ProgressBar():
            ds = (
                ds.assign_coords(x=ds.x, y=ds.y)
                .chunk({"level": 1})
                .drop_vars(["valid_time", "step"])
            )
            ds.to_zarr(
                path,
                consolidated=True,
                mode="w",
                encoding={
                    var: {"compressor": compressor} for var in ds.data_vars
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process grib2 files and save them to zarr format."
    )
    parser.add_argument(
        "--data_in",
        type=str,
        default="/scratch/mch/dealmeih/kenda/",
        help="Input data directory.",
    )
    parser.add_argument(
        "--data_train",
        type=str,
        default="/scratch/mch/sadamov/train.zarr",
        help="Output zarr file for training.",
    )
    parser.add_argument(
        "--data_test",
        type=str,
        default="/scratch/mch/sadamov/test.zarr",
        help="Output zarr file for testing.",
    )
    parser.add_argument(
        "--indexpath",
        type=str,
        default="/scratch/mch/sadamov/temp",
        help="Path to the index file.",
    )

    SELECTED_VARS = {
        "heightAboveGround": ["U_10M", "V_10M"],
        "surface": ["PS", "HSURF", "TQV", "FIS", "CLCL"],
        "hybridLayer": ["T", "pp", "QV"],
        "hybrid": ["W"],
        "meanSea": ["PMSL"],
    }
    SELECTED_VARS_2 = {
        # U,V have different lat/lon. T_2M has different heightAboveGround
        "shortName": ["T_2M", "U", "V"],
        # TOT_PREC retrieval incorrect in analysis file
    }

    args = parser.parse_args()

    main(
        args.data_in,
        args.data_train,
        args.data_test,
        args.indexpath,
        SELECTED_VARS,
        SELECTED_VARS_2,
    )
