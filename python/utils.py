import json
import numpy as np
import pptk


def read_pointcloud(filename):
    """Reads a pointcloud from a file into a JSON

    """

    with open(filename) as f:
        return json.load(f)


def pointcloud_to_numpy_array(pointcloud):
    """Converts a JSON pointcloud to an Nx4 (x,y,z,intensity) numpy matrix for easier manipulation

    """

    array = np.zeros((len(pointcloud), 4), dtype=np.float)

    for i, point in enumerate(pointcloud):
        array[i] = [point["x"], point["y"], point["z"], point["intensity"]]

    return array


def show(points):
    """Shows an Nx4 numpy matrix in 3D, using the intensity for a color scale

    """
    v = pptk.viewer((points[:, :3]), points[:, 3])
    v.set(point_size=0.02)
