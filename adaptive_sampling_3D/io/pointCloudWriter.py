import os

import numpy as np
import open3d as o3d
import zarr

# Sub-group, inside an OME-Zarr structure, holding this pipeline's point clouds.
# Mirrors the OME-Zarr ``labels/<name>`` convention for derived data.
POINTS_GROUP = "points"
POINTS_NAME = "adaptive_sampling_3D"


def save_pcd(input_pcd, result_path):

    try:
        directory = os.path.dirname(result_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        success = o3d.io.write_point_cloud(result_path, input_pcd)
        return success

    except Exception as e:
        print(f"Error: {e}")
        return False


def save_pcd_ome_zarr(input_pcd, zarr_root, name=POINTS_NAME):
    """Write a colored point cloud into an OME-Zarr structure.

    The point cloud is stored as a Zarr v3 (OME-Zarr v0.5) sub-group at
    ``<zarr_root>/points/<name>`` holding two arrays:

    * ``coordinates`` -- ``(N, 3)`` float32 point positions in ``(z, y, x)`` order.
    * ``colors``      -- ``(N, 3)`` float32 RGB colors in ``[0, 1]`` (omitted when absent).

    Args:
        input_pcd (o3d.geometry.PointCloud): Point cloud to persist.
        zarr_root (str): Path to the OME-Zarr group that will contain the ``points`` sub-group.
        name (str, optional): Name of the point set under ``points/``. Defaults to ``adaptive_sampling_3D``.

    Returns:
        bool: ``True`` on success, ``False`` otherwise.
    """

    try:
        coordinates = np.asarray(input_pcd.points, dtype=np.float32)
        colors = np.asarray(input_pcd.colors, dtype=np.float32)

        # Open (or create) the OME-Zarr root group and the nested points sub-group
        root = zarr.open_group(store=zarr_root, mode="a", zarr_format=3)
        points_group = root.require_group(POINTS_GROUP).require_group(name)

        n_points = coordinates.shape[0]

        # Point positions; a single chunk keeps clouds contiguous for downstream reads
        coord_array = points_group.create_array(
            "coordinates",
            shape=coordinates.shape,
            dtype=coordinates.dtype,
            chunks=coordinates.shape if n_points else (1, 3),
            overwrite=True,
        )
        if n_points:
            coord_array[:] = coordinates

        attributes = ["coordinates"]

        # Per-point colors, when present
        if colors.size:
            color_array = points_group.create_array(
                "colors",
                shape=colors.shape,
                dtype=colors.dtype,
                chunks=colors.shape,
                overwrite=True,
            )
            color_array[:] = colors
            attributes.append("colors")

        # OME-Zarr styled metadata describing the point set
        points_group.attrs["ome"] = {"version": "0.5"}
        points_group.attrs["points"] = {
            "version": "0.1",
            "count": int(n_points),
            "axes": ["z", "y", "x"],
            "attributes": attributes,
        }

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False
