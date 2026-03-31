import torch

def lat_lon_to_cartesian(lat, lon):
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    return torch.stack([x, y, z], dim=-1)


def get_area_weights(lat):
    return torch.cos(torch.deg2rad(lat))

def get_cartesian_coords(self):
    lat, lon = self.get_lat_lon()
    return lat_lon_to_cartesian(lat, lon)