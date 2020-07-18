import numpy as np

def x(lon, z):
    return 1/360 * 2 ** z * (lon + 180)

def y(lat, z):
    rads = lat * (2 * np.pi) / 360
    return 2 ** z * (np.pi - np.log(np.tan(np.pi / 4 + rads / 2))) / (2 * np.pi)

def xy(lat, lon, z):
    return x(lon, z), y(lat, z)

def lon(x, z):
    return 360 * x / 2 ** z - 180

def lat(y, z):
    rads = 2 * (np.arctan(np.exp(np.pi - 2 * np.pi * y / 2 ** z)) - np.pi/4)
    return 360 * rads / (2 * np.pi)

def latlon(x, y, z):
    return lat(y,z), lon(x,z)

