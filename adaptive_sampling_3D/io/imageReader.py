"""Format-agnostic image reading interface.

Dispatches the reading primitives used by the point cloud pipeline to a
backend selected from the input path:

* HDF5 (``.h5``/``.hdf5`` files)     -> :mod:`hdf5Reader`
* OME-Zarr v0.5 (``.zarr`` groups)   -> :mod:`omeZarrReader`

All public functions keep the same signatures across backends so callers do
not need to know which storage format is in use.
"""

import os

from adaptive_sampling_3D.io import hdf5Reader
from adaptive_sampling_3D.io import omeZarrReader

# Recognised input formats keyed by lower-cased extension / suffix
HDF5_SUFFIXES = (".h5", ".hdf5")
ZARR_SUFFIXES = (".zarr",)


def is_ome_zarr(img_path):
    """Return ``True`` when ``img_path`` points to an OME-Zarr group.

    An OME-Zarr dataset is a directory that either carries a ``.zarr`` suffix
    (``foo.ome.zarr`` included) or contains Zarr group metadata.
    """

    path = img_path.rstrip("/")
    lowered = path.lower()

    if lowered.endswith(ZARR_SUFFIXES):
        return True

    if os.path.isdir(path):
        # Zarr v3 stores group metadata in zarr.json, v2 in .zgroup
        return os.path.exists(os.path.join(path, "zarr.json")) or \
            os.path.exists(os.path.join(path, ".zgroup"))

    return False


def is_hdf5(img_path):
    """Return ``True`` when ``img_path`` points to an HDF5 file."""

    return img_path.lower().endswith(HDF5_SUFFIXES)


def _backend(img_path):
    """Select the reader backend module for a given input path."""

    if is_ome_zarr(img_path):
        return omeZarrReader
    if is_hdf5(img_path):
        return hdf5Reader

    raise ValueError(
        f"Unsupported image format for '{img_path}'. "
        f"Expected an HDF5 file ({', '.join(HDF5_SUFFIXES)}) "
        f"or an OME-Zarr group ({', '.join(ZARR_SUFFIXES)})."
    )


def get_channels(img_path):
    """Retrieve the list of channel identifiers from an image.

    Args:
        img_path (str): Path to the image (HDF5 file or OME-Zarr group).

    Returns:
        list: Channel identifiers (dataset names for HDF5, channel indices for OME-Zarr).
    """

    return _backend(img_path).get_channels(img_path)


def iter_tiles(img_path, block_size):
    """Iterate over non-overlapping 3D tiles of an image volume.

    Args:
        img_path (str): Path to the image (HDF5 file or OME-Zarr group).
        block_size (tuple of int): Tile size ``(Z, Y, X)``.

    Yields:
        dict: ``{"cuboid": (Z, Y, X, C) ndarray, "index": [z, y, x], "shape": (dz, dy, dx)}``.
    """

    return _backend(img_path).iter_tiles(img_path, block_size)


def get_indices(img_path, block_size):
    """Generate non-overlapping 3D coordinates and shapes for volume tiling.

    Args:
        img_path (str): Path to the image (HDF5 file or OME-Zarr group).
        block_size (tuple): Target tile size ``(z, y, x)``.

    Yields:
        tuple: ``(index, shape)`` with ``index`` as ``[z, y, x]`` and ``shape`` as ``[dz, dy, dx]``.
    """

    return _backend(img_path).get_indices(img_path, block_size)


def get_tile(img_path, index, shape):
    """Slice a 3D multi-channel tile volume from an image.

    Args:
        img_path (str): Path to the image (HDF5 file or OME-Zarr group).
        index (tuple/list): Starting ``[z, y, x]`` coordinates of the tile.
        shape (tuple/list): Dimensions ``[dz, dy, dx]`` of the tile to extract.

    Returns:
        np.ndarray: A 4D array of shape ``(dz, dy, dx, C)`` with the stacked channel data.
    """

    return _backend(img_path).get_tile(img_path, index, shape)


def sample_volume_histograms(img_path, channel, n_samples=10000):
    """Estimate the intensity distribution of a channel by random voxel sampling.

    Args:
        img_path (str): Path to the image (HDF5 file or OME-Zarr group).
        channel: Channel identifier as returned by :func:`get_channels`.
        n_samples (int, optional): Number of random voxels to sample. Defaults to 10000.

    Returns:
        np.ndarray: A 1D array of length ``n_samples`` with the sampled intensity values.
    """

    return _backend(img_path).sample_volume_histograms(img_path, channel, n_samples)
