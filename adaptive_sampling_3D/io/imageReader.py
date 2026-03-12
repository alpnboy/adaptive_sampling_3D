import h5py as h5
import numpy as np

def get_channels(img_path):
    """ Retrieves the list of channel names from an HDF5 file.

    Args:
        h5_path (str): Full path to the .h5 file.

    Returns:
        list[str]: A list of strings containing the names of the datasets (channels).
    """
    
    with h5.File(img_path, "r") as f:
        channels = list(f.keys())
        
    return channels


def iter_tiles(img_path, block_size):
    """ Generetator function. Iterates over 3D tiles in a H5DF dataset.
    
    This function iteratively return non overlapping chunks of the image volume with size defined by "block_size".
    
    Args:
    img_path (str): Path to the H5DF file
    
    block_size (tuple of int): Defines the tile size (Z, Y, X)
    
    Yelds:
        dict: Dictionary contaning:
            cuboid (np.ndarray): Tile data (Z, Y, X, C) intensities (uint16)
            index (tuple of int): Starting coordinates (Z, Y, X)
            shape (tuple of int): Tile shape 
    """
    
    bz, by, bx = block_size
    
    f = h5.File(img_path, "r")
    try:
        keys = list(f.keys())
        d0 = f[keys[0]]
        d1 = f[keys[1]]

        Z, Y, X = d0.shape

        for z in range(0, Z, bz):
            z1 = min(z + bz, Z)

            for y in range(0, Y, by):
                y1 = min(y + by, Y)

                for x in range(0, X, bx):
                    x1 = min(x + bx, X)

                    c0 = d0[z:z1, y:y1, x:x1]
                    c1 = d1[z:z1, y:y1, x:x1]

                    cuboid = np.stack((c0, c1), axis=-1)

                    yield {"cuboid": cuboid, "index": [z, y, x], "shape": (z1 - z, y1 - y, x1 - x),}
    finally:
        f.close()

        
def get_indices(img_path, block_size):
    """ Generates non-overlapping 3D coordinates and shapes for volume tiling.

    Args:
        img_path (str): Path to the H5 file to get global dimensions.
        block_size (tuple): Target 3D tile size (z, y, x).

    Yields:
        tuple: (index, shape) with index is [z, y, x] and shape [dz, dy, dx].
    """
    
    bz, by, bx = block_size
    
    f = h5.File(img_path, "r")
    try:
        keys = list(f.keys())
        d0 = f[keys[0]]
        d1 = f[keys[1]]

        Z, Y, X = d0.shape

        for z in range(0, Z, bz):
            z1 = min(z + bz, Z)

            for y in range(0, Y, by):
                y1 = min(y + by, Y)

                for x in range(0, X, bx):
                    x1 = min(x + bx, X)
                    
                    index = [z, y, x]
                    shape = [z1 - z, y1 - y, x1 - x]
                    
                    yield index, shape

    finally:
        f.close()
        
        
def get_tile(img_path, index, shape):
    """ Slices a 3D multi-channel tile volume from an HDF5 image.

    Args:
        img_path (str): Path to the HDF5 image file.
        index (tuple/list): Starting [z, y, x] coordinates of the tile.
        shape (tuple/list): Dimensions [dz, dy, dx] of the tile to extract.

    Returns:
        np.ndarray: A 4D array of shape (dz, dy, dx, 2) containing 
            the stacked channel data.
    """
    
    # Unpack tile start coordinates and dimension
    z, y, x = index
    bz, by, bx = shape
    
    # Open the h5 image and slices the channel datasets
    with h5.File(img_path, "r") as f:
        keys = list(f.keys())
        d0 = f[keys[0]][z:z+bz, y:y+by, x:x+bx]
        d1 = f[keys[1]][z:z+bz, y:y+by, x:x+bx]
        
    # Stack the two channels
    tile_volume = np.stack((d0, d1), axis=-1)    

    return tile_volume

