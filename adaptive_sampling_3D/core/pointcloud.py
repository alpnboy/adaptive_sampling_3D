import numpy as np
import open3d as o3d

def translate_pcd_coordinates(pcd, offset_coordinates):
    """ Applies a spatial translation to a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to translate.
        offset_coordinates (tuple/list): [Z, Y, X] offset vector for translation.

    Returns:
        o3d.geometry.PointCloud: Pointcloud with new coordinates (Z, Y, X) + offset.
    """
    
    pcd.translate(offset_coordinates)
    
    return pcd

def merge_tile_pcd(merged_pcd, tile_pcd, offset_coordinates):
    """ Merges a new tile point cloud into a global point cloud.
    
    Args:
        merged_pcd (o3d.geometry.PointCloud): The global destination point cloud.
        tile_pcd (o3d.geometry.PointCloud): The local tile to be added.
        offset_coordinates (list): [Z, Y, X] origin of the tile in the global reference cloud.

    Returns:
        o3d.geometry.PointCloud: The updated global point cloud.
    """
    
    # Copy tile point cloud and translate coordinates to absolute global reference
    tile_copy = tile_pcd.translate(offset_coordinates, relative=True)
    
    # Merge points to target pointcloud
    merged_pcd += tile_copy
    
    return merged_pcd

def colored_pcd(pcd, volume):
    """ Generates colored point cloud by mapping original volume intensities into cloud space.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud (Z, Y, X) geometry to color.
        volume (np.ndarray): 4D normalized volume (Z, Y, X, Channels).

    Returns:
        o3d.geometry.PointCloud: A new point cloud with RGB colors derived 
            from volume intensities.
    """

    # Convert pcd.points into array coordinates
    intensities = np.zeros([len(pcd.points), 3])
    coordinates = np.asarray(pcd.points)
    coordinates = coordinates.astype(int)
    
    # Use coordinate to retrieve the original intensities from the normalized volume
    intensities[:,0:2] = volume[coordinates[:,0], coordinates[:,1], coordinates[:,2], 0:2]
    
    # Store intensities into colored point cloud
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(coordinates)
    pcd_colored.colors = o3d.utility.Vector3dVector(intensities)
    
    return pcd_colored
    
