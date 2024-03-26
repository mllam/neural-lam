"""unrotate rotated pole coordinates to geographical lat/lon"""

# Third-party
import numpy as np

# First-party
from neural_lam import constants


def unrot_lon(rotlon, rotlat, pollon, pollat):
    """Transform rotated longitude to longitude.

    Parameters
    ----------
    rotlon : np.ndarray(i,j)
        rotated longitude (deg)
    rotlat : np.ndarray(i,j)
        rotated latitude (deg)
    pollon : float
        rotated pole longitude (deg)
    pollat : float
        rotated pole latitude (deg)

    Returns
    -------
    lon : np.ndarray(i,j)
        geographical longitude

    """

    # to radians
    rlo = np.radians(rotlon)
    rla = np.radians(rotlat)

    # sin and cos of pole position
    s1 = np.sin(np.radians(pollat))
    c1 = np.cos(np.radians(pollat))
    s2 = np.sin(np.radians(pollon))
    c2 = np.cos(np.radians(pollon))

    # subresults
    tmp1 = s2 * (
        -s1 * np.cos(rlo) * np.cos(rla) + c1 * np.sin(rla)
    ) - c2 * np.sin(rlo) * np.cos(rla)
    tmp2 = c2 * (
        -s1 * np.cos(rlo) * np.cos(rla) + c1 * np.sin(rla)
    ) + s2 * np.sin(rlo) * np.cos(rla)

    return np.degrees(np.arctan(tmp1 / tmp2))


def unrot_lat(rotlat, rotlon, pollat):
    """Transform rotated latitude to latitude.

    Parameters
    ----------
    rotlat : np.ndarray(i,j)
        rotated latitude (deg)
    rotlon : np.ndarray(i,j)
        rotated longitude (deg)
    pollon : float
        rotated pole longitude (deg)
    pollat : float
        rotated pole latitude (deg)

    Returns
    -------
    lat : np.ndarray(i,j)
        geographical latitude

    """

    # to radians
    rlo = np.radians(rotlon)
    rla = np.radians(rotlat)

    # sin and cos of pole position
    s1 = np.sin(np.radians(pollat))
    c1 = np.cos(np.radians(pollat))

    # subresults
    tmp1 = s1 * np.sin(rla) + c1 * np.cos(rla) * np.cos(rlo)

    return np.degrees(np.arcsin(tmp1))


def unrotate_latlon(data):
    """Unrotate lat/lon coordinates from rotated pole grid."""
    xx, yy = np.meshgrid(data.x.values, data.y.values)
    # unrotate lon/lat
    lon = unrot_lon(xx, yy, constants.POLLON, constants.POLLAT)
    lat = unrot_lat(yy, xx, constants.POLLAT)

    return lon.T, lat.T
