"""OME-Zarr v0.5 image reader.

Provides the same reading interface as :mod:`hdf5Reader` (``get_channels``,
``iter_tiles``, ``get_indices``, ``get_tile`` and ``sample_volume_histograms``)
backed by the ``ome-zarr-py`` package so that the point cloud pipeline can
consume OME-Zarr v0.5 (Zarr v3) image volumes transparently.

OME-Zarr stores every channel inside a single multi-dimensional array whose
axis order is described by the ``multiscales`` metadata (typically ``czyx`` or
``tczyx``). This module resolves that axis mapping and exposes per-channel
``(Z, Y, X)`` volumes stacked on the last axis, matching the ``(Z, Y, X, C)``
layout produced by the HDF5 reader.
"""

import logging

import numpy as np

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

logger = logging.getLogger(__name__)

# The image is re-opened per tile, so silence ome-zarr's verbose per-open INFO logs
logging.getLogger("ome_zarr").setLevel(logging.WARNING)

SUPPORTED_VERSION = "0.5"


def _open_image(img_path):
    """Open an OME-Zarr image and return its highest resolution array and axes.

    Args:
        img_path (str): Path (or URL) to the OME-Zarr group.

    Returns:
        tuple: ``(data, axes)`` where ``data`` is a dask array of the full
            resolution level and ``axes`` is the ordered list of axis names
            (e.g. ``['c', 'z', 'y', 'x']``).
    """

    location = parse_url(img_path, mode="r")
    if location is None:
        raise ValueError(f"'{img_path}' is not a readable OME-Zarr group")

    # Warn (but do not fail) when the on-disk version is not the targeted v0.5
    version = location.root_attrs.get("version") or getattr(location.fmt, "version", None)
    if version is not None and str(version) != SUPPORTED_VERSION:
        logger.warning(
            "OME-Zarr version '%s' found, this reader targets v%s; proceeding best-effort",
            version, SUPPORTED_VERSION,
        )

    node = next(iter(Reader(location)()), None)
    if node is None or not node.data:
        raise ValueError(f"No multiscale image data found in '{img_path}'")

    axes = [axis["name"] for axis in node.metadata["axes"]]

    # Highest resolution level is always the first dataset in the multiscale list
    data = node.data[0]

    return data, axes


def _axis_index(axes, name):
    """Return the position of a named axis, or ``None`` when it is absent."""

    return axes.index(name) if name in axes else None


def _spatial_shape(data, axes):
    """Extract the ``(Z, Y, X)`` volume shape from an OME-Zarr array."""

    zi, yi, xi = _axis_index(axes, "z"), _axis_index(axes, "y"), _axis_index(axes, "x")
    if None in (zi, yi, xi):
        raise ValueError(f"OME-Zarr image is missing spatial axes, found axes: {axes}")

    return data.shape[zi], data.shape[yi], data.shape[xi]


def _num_channels(data, axes):
    """Return the number of channels in the OME-Zarr image (1 if no channel axis)."""

    ci = _axis_index(axes, "c")
    return data.shape[ci] if ci is not None else 1


def _channel_volume(data, axes, channel):
    """Return a ``(Z, Y, X)`` dask view for a single channel of the image.

    Reduces the time axis (first index) and the channel axis (selected index),
    then transposes the remaining spatial axes into ``(z, y, x)`` order.
    """

    selector = []
    for axis in axes:
        if axis == "t":
            selector.append(0)
        elif axis == "c":
            selector.append(channel)
        else:
            selector.append(slice(None))

    volume = data[tuple(selector)]

    # Remaining axes after dropping scalar t/c selections, in file order
    spatial_axes = [axis for axis in axes if axis not in ("t", "c")]
    order = [spatial_axes.index(axis) for axis in ("z", "y", "x")]

    return volume.transpose(order)


def get_channels(img_path):
    """Retrieve the list of channel indices from an OME-Zarr image.

    Channels live on the ``c`` axis of a single array, so they are returned as
    integer indices (``[0, 1, ...]``) that address that axis.

    Args:
        img_path (str): Path to the OME-Zarr group.

    Returns:
        list[int]: One index per channel present in the image.
    """

    data, axes = _open_image(img_path)
    return list(range(_num_channels(data, axes)))


def _tile_bounds(shape, block_size):
    """Yield ``(index, shape)`` pairs covering ``shape`` with ``block_size`` tiles."""

    Z, Y, X = shape
    bz, by, bx = block_size

    for z in range(0, Z, bz):
        z1 = min(z + bz, Z)
        for y in range(0, Y, by):
            y1 = min(y + by, Y)
            for x in range(0, X, bx):
                x1 = min(x + bx, X)
                yield [z, y, x], [z1 - z, y1 - y, x1 - x]


def _stack_tile(data, axes, index, shape):
    """Slice and stack all channels of a tile into a ``(dz, dy, dx, C)`` array."""

    z, y, x = index
    dz, dy, dx = shape

    selector = []
    for axis in axes:
        if axis == "t":
            selector.append(0)
        elif axis == "c":
            selector.append(slice(None))
        elif axis == "z":
            selector.append(slice(z, z + dz))
        elif axis == "y":
            selector.append(slice(y, y + dy))
        elif axis == "x":
            selector.append(slice(x, x + dx))

    block = np.asarray(data[tuple(selector)])

    # Transpose to (z, y, x, c); a missing channel axis becomes a trailing size-1 axis
    remaining = [axis for axis in axes if axis != "t"]
    if "c" in remaining:
        order = [remaining.index(axis) for axis in ("z", "y", "x", "c")]
        return np.transpose(block, order)

    order = [remaining.index(axis) for axis in ("z", "y", "x")]
    return np.transpose(block, order)[..., np.newaxis]


def iter_tiles(img_path, block_size):
    """Iterate over non-overlapping 3D tiles of an OME-Zarr image volume.

    Args:
        img_path (str): Path to the OME-Zarr group.
        block_size (tuple of int): Tile size ``(Z, Y, X)``.

    Yields:
        dict: ``{"cuboid": (Z, Y, X, C) ndarray, "index": [z, y, x], "shape": (dz, dy, dx)}``.
    """

    data, axes = _open_image(img_path)
    shape = _spatial_shape(data, axes)

    for index, tile_shape in _tile_bounds(shape, block_size):
        cuboid = _stack_tile(data, axes, index, tile_shape)
        yield {"cuboid": cuboid, "index": index, "shape": tuple(tile_shape)}


def get_indices(img_path, block_size):
    """Generate non-overlapping 3D coordinates and shapes for volume tiling.

    Args:
        img_path (str): Path to the OME-Zarr group.
        block_size (tuple): Target tile size ``(z, y, x)``.

    Yields:
        tuple: ``(index, shape)`` with ``index`` as ``[z, y, x]`` and ``shape`` as ``[dz, dy, dx]``.
    """

    data, axes = _open_image(img_path)
    shape = _spatial_shape(data, axes)

    for index, tile_shape in _tile_bounds(shape, block_size):
        yield index, tile_shape


def get_tile(img_path, index, shape):
    """Slice a 3D multi-channel tile volume from an OME-Zarr image.

    Args:
        img_path (str): Path to the OME-Zarr group.
        index (tuple/list): Starting ``[z, y, x]`` coordinates of the tile.
        shape (tuple/list): Dimensions ``[dz, dy, dx]`` of the tile to extract.

    Returns:
        np.ndarray: A 4D array of shape ``(dz, dy, dx, C)`` with the stacked channel data.
    """

    data, axes = _open_image(img_path)
    return _stack_tile(data, axes, index, shape)


def sample_volume_histograms(img_path, channel, n_samples=10000):
    """Estimate the intensity distribution of a channel by random voxel sampling.

    Args:
        img_path (str): Path to the OME-Zarr group.
        channel (int): Index of the channel (position on the ``c`` axis) to sample.
        n_samples (int, optional): Number of random voxels to sample. Defaults to 10000.

    Returns:
        np.ndarray: A 1D array of length ``n_samples`` with the sampled intensity values.
    """

    data, axes = _open_image(img_path)
    volume = _channel_volume(data, axes, channel)

    Z, Y, X = volume.shape
    coordinates = np.random.randint([0, 0, 0], (Z, Y, X), size=(n_samples, 3))

    # Vectorised point read across chunks, then materialised to a numpy array
    sampled_values = volume.vindex[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]

    return np.asarray(sampled_values)
