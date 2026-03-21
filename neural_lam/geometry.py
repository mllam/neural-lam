import numpy as np

def lat_lon_to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Convert (lon, lat) in degrees to (x, y, z) on unit sphere."""
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)

def get_area_weights(lat: np.ndarray) -> np.ndarray:
    """Return cos(latitude) weights normalized by mean weight."""
    weights = np.cos(np.radians(lat))
    return weights / weights.mean()
