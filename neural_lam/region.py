# Standard library
from dataclasses import dataclass

# Third-party
import numpy as np
import yaml


@dataclass
class RegionConfig:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    resolution: float
    projection: str = "latlon"

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def generate_xy_from_region(region: RegionConfig):
    lats = np.arange(region.min_lat, region.max_lat, region.resolution)

    lons = np.arange(region.min_lon, region.max_lon, region.resolution)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    xy = np.stack((lon_grid, lat_grid), axis=-1)
    return xy
