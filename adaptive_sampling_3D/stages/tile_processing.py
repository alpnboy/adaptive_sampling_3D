import numpy as np
import open3d as o3d

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from adaptive_sampling_3D.io import read
from adaptive_sampling_3D.core import volume as vlm
from adaptive_sampling_3D.core import mesh 
from adaptive_sampling_3D.core import pointcloud as pcd

def process_tile(img_path, index, shape, percentiles, backgrounds, channels):
    """ Orchestrates processing of individual volume tiles into colored point cloud tiles.

    Args:
        img_path (_type_): _description_
        index (list): _description_
        shape (_type_): _description_
        percentiles (float): _description_
        backgrounds (float): _description_
        channels (list): list of tuples containing per channel point cloud and positional index

    Returns:
        _type_: _description_
    """
    
    # Float conversion to avoid uint16 integer underflow
    tile_volume = read.get_tile(img_path, index, shape).astype(np.float32)
        
    # Background subtraction and normalization based on estimate from whole volume
    for ch in range(0,len(channels)):
        tile_volume[:,:,:,ch] = vlm.volume_background_subtraction(tile_volume[:,:,:,ch], backgrounds[ch])
        tile_volume[:,:,:,ch] = vlm.volume_normalization(tile_volume[:,:,:,ch], percentiles[ch])  
          
    results = []
         
    # Generate mesh from separate channels    
    for ch in range(1, 2):              # This is hardcoded, for now I only use the second channel with the nuclei information
        v_ch = tile_volume[:,:,:,ch]
        t_mesh = vlm.volume_to_mesh_marching_cubes(v_ch, 0.8)
        if t_mesh == []: continue
            
        # Estimate density of tile cloud based on mesh area
        n_points = np.min([10000,int(mesh.estimate_npoints_sigmoid(t_mesh, 30000))]) 

        # Convert mesh to pcd and retrieve original intensities
        t_pcd = mesh.mesh_to_cloud_uniform(t_mesh, n_points)
        pcd_colored = pcd.colored_pcd(t_pcd, tile_volume)
        
        results.append((pcd_colored, index))
                
    return results

def process_tiles_sequential(img_path, percentiles, backgrounds, channels, block_size = (500, 500, 500)):
    
    tile_n = 0
    merged_pcd = o3d.geometry.PointCloud()
    
    for iter in read.iter_tiles(img_path, block_size):
        tile_n += 1
        print("tile number", tile_n)
        tile_volume = iter.get("cuboid").astype(np.float32)
        index = iter.get("index")        
        
        for ch in range(0,len(channels)):
            tile_volume[:,:,:,ch] = vlm.volume_background_subtraction(tile_volume[:,:,:,ch], backgrounds[ch])
            tile_volume[:,:,:,ch] = vlm.volume_normalization(tile_volume[:,:,:,ch], percentiles[ch])    
        
        for ch in range(1, 2):
            v_ch = tile_volume[:,:,:,ch]
            t_mesh = vlm.volume_to_mesh_marching_cubes(v_ch, 0.8)
            
            if t_mesh == []: continue
            
            n = np.min([10000,int(mesh.estimate_npoints_sigmoid(t_mesh, 30000))]) 
            t_pcd = mesh.mesh_to_cloud_uniform(t_mesh, n)
            pcd_colored = pcd.colored_pcd(t_pcd, tile_volume)
            merged_pcd = pcd.merge_tile_pcd(merged_pcd, pcd_colored, index)
            
    return merged_pcd


def process_tiles_multi_threaded(img_path, percentiles, backgrounds, channels, block_size = [500, 500, 500], max_workers = 8):
    
    # Pre-compute all tile indices for workers
    all_tiles = list(read.get_indices(img_path, block_size))
    total_indices = len(all_tiles)
    
    # List for processed pointcloud
    tiles_pcd = []                  # Change to preallocated list with indexes for speed
    
    # Initialize pointcloud to merge tiled clouds
    merged_pcd = o3d.geometry.PointCloud()

    if total_indices == 0:
        print("Warning: No tiles found.")
        return merged_pcd

    # Pre initialize all arguments
    args = [
        (img_path, index, shape, percentiles, backgrounds, channels)
        for index, shape in all_tiles
    ]
    
    # Multi threading by mapping arguments
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        
        for tile_results in tqdm(executor.map(lambda a: process_tile(*a), args), total = total_indices):
            
            # Clouds are appended in main thread
            for pcd_colored, index in tile_results:
                tiles_pcd.append((pcd_colored, index))
                merged_pcd = pcd.merge_tile_pcd(merged_pcd, pcd_colored, index)
    
    #for pcd_colored, index in tiles_pcd:            
    #    merged_pcd = pcd.merge_tile_pcd(merged_pcd, pcd_colored, index)

    return merged_pcd