# Standard library
import functools
from pathlib import Path

# Third-party
import cartopy.crs as ccrs
import yaml


class Config:
    """
    Class for loading configuration files.

    This class loads a configuration file and provides a way to access its
    values as attributes.
    """

    def __init__(self, values):
        self.values = values

    @classmethod
    def from_file(cls, filepath):
        """Load a configuration file."""
        if filepath.endswith(".yaml"):
            with open(filepath, encoding="utf-8", mode="r") as file:
                return cls(values=yaml.safe_load(file))
        else:
            raise NotImplementedError(Path(filepath).suffix)

    def __getattr__(self, name):
        keys = name.split(".")
        value = self.values
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return None
        if isinstance(value, dict):
            return Config(values=value)
        return value

    def __getitem__(self, key):
        value = self.values[key]
        if isinstance(value, dict):
            return Config(values=value)
        return value

    def __contains__(self, key):
        return key in self.values

    def num_data_vars(self):
        """Return the number of data variables for a given key."""
        return len(self.dataset.var_names)

    @functools.cached_property
    def coords_projection(self):
        """Return the projection."""
        proj_config = self.values["projection"]
        proj_class_name = proj_config["class"]
        proj_class = getattr(ccrs, proj_class_name)
        proj_params = proj_config.get("kwargs", {})
        return proj_class(**proj_params)
