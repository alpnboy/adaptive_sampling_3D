import os
import argparse

from adaptive_sampling_3D.io import read
from adaptive_sampling_3D.io import write as wrt
from adaptive_sampling_3D.stages import normalization as norm
from adaptive_sampling_3D.stages import tile_processing as tiling


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Generate a dataset of colored pointclouds by adaptive sampling large 3D images')
    parser.add_argument('--input_dir', type = str, default = './data/raw_h5', help = 'directory with input volume images')
    parser.add_argument('--result_dir', type = str, default = './data/results', help = 'directory where point clouds are saved')
    parser.add_argument('--multi_thread', type = bool, default = True, help = 'thread processing of volume tiles')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of workers for multi threading (default = 8)')
    parser.add_argument('--block_size', type = int, nargs = 3, default = [500, 500, 500], help = 'size of the volume tiles (z, y, x)')
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    result_dir = os.path.abspath(args.result_dir)
    multi_thread = args.multi_thread
    num_workers = args.num_workers
    block_size = args.block_size
    
    # Check result folder exists
    os.makedirs(result_dir, exist_ok=True)
    
    # Only grabs h5 files
    file_names = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    
    for file_name in file_names:
        
        # Path of target h5 image and resulting pcd
        img_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0]
        result_path = os.path.join(result_dir, f"{base_name}.ply")
        
        # Compute normalization parameters from whole slide image
        channels = read.get_channels(img_path)
        percentiles, backgrounds = norm.compute_slide_normalization(img_path, channels)
        
        if multi_thread == True:
            # Parallel processing of volume tiles to pcd
            merged_pcd = tiling.process_tiles_multi_threaded(img_path, percentiles, backgrounds, channels, block_size = block_size, max_workers = num_workers)
        else:
            # Sequential processing of volume tiles to pcd
            merged_pcd = tiling.process_tiles_sequential(img_path, percentiles, backgrounds, channels, block_size = block_size)
            
        # Save global point cloud
        wrt.save_pcd(merged_pcd, result_path)
                    
if __name__ == "__main__":
    main()



