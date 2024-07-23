# Standard library
from pathlib import Path

# Third-party
import yaml


class Config:
    """Class to load and access the configuration file."""

    def __init__(self, values):
        self.values = values

    @classmethod
    def from_file(cls, filepath):
        """Load the configuration file from the given path."""
        if filepath.endswith(".yaml"):
            with open(filepath, encoding="utf-8", mode="r") as file:
                return cls(values=yaml.safe_load(file))
        else:
            raise NotImplementedError(Path(filepath).suffix)

    def __getattr__(self, name):
        """Recursively access the values in the configuration."""
        keys = name.split(".")
        value = self.values
        for key in keys:
            try:
                value = value[key]
            except KeyError:
                raise AttributeError(f"Key '{key}' not found in {value}")
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
