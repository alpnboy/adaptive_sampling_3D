import os
import argparse
import logging
import sys

from adaptive_sampling_3D.io import imageReader
from adaptive_sampling_3D.io import pointCloudWriter
from adaptive_sampling_3D.stages import normalization as norm
from adaptive_sampling_3D.stages import tile_processing as tiling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Logger config for calling as module
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

def _strip_image_suffix(file_name):
    """Return the base name of an image, dropping known image suffixes.

    Handles ``.h5`` files as well as ``.zarr`` / ``.ome.zarr`` groups, e.g.
    ``sample.ome.zarr`` -> ``sample`` and ``sample.h5`` -> ``sample``.
    """

    base = file_name.rstrip('/')
    lowered = base.lower()

    if lowered.endswith('.zarr'):
        base = base[:-len('.zarr')]
        if base.lower().endswith('.ome'):
            base = base[:-len('.ome')]
        return base

    return os.path.splitext(base)[0]


def _output_formats(output_format, is_ome_zarr):
    """Return the point cloud formats to write for one input image.

    The point cloud itself carries no trace of the container it was read from,
    so any input format can be written to any output format. ``auto`` keeps the
    output matched to the input.

    Args:
        output_format (str): Value of ``--output_format``.
        is_ome_zarr (bool): Whether the input image is an OME-Zarr group.

    Returns:
        tuple[str, ...]: Some combination of ``'ply'`` and ``'zarr'``.
    """

    if output_format == 'auto':
        return ('zarr',) if is_ome_zarr else ('ply',)

    if output_format == 'both':
        return ('zarr', 'ply')

    return (output_format,)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Generate a dataset of colored pointclouds by adaptive sampling large 3D images')
    parser.add_argument('--input_dir', type = str, default = './data/raw_h5', help = 'directory with input volume images')
    parser.add_argument('--result_dir', type = str, default = './data/results', help = 'directory where point clouds are saved')
    parser.add_argument('--multi_thread', type = bool, default = True, help = 'thread processing of volume tiles')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of workers for multi threading (default = 8)')
    parser.add_argument('--block_size', type = int, nargs = 3, default = [500, 500, 500], help = 'size of the volume tiles (z, y, x)')
    parser.add_argument('--output_format', type = str, default = 'auto', choices = ['auto', 'ply', 'zarr', 'both'], help = "point cloud output format ('auto' follows the input: PLY for HDF5, OME-Zarr for OME-Zarr)")
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    result_dir = os.path.abspath(args.result_dir)
    multi_thread = args.multi_thread
    output_format = args.output_format
    num_workers = args.num_workers
    block_size = args.block_size
    
    # Check result folder exists
    os.makedirs(result_dir, exist_ok=True)

    # Grabs HDF5 files and OME-Zarr groups (directories ending in .zarr, .ome.zarr included)
    file_names = [
        f for f in os.listdir(input_dir)
        if f.endswith('.h5') or f.rstrip('/').lower().endswith('.zarr')
    ]

    logger.info(f"Processing files: {file_names}")

    for file_name in file_names:
        logger.info(f"Processing file: {file_name}")
        # Path of target image and base name for the result
        img_path = os.path.join(input_dir, file_name)
        is_ome_zarr = imageReader.is_ome_zarr(img_path)
        base_name = _strip_image_suffix(file_name)

        # Compute normalization parameters from whole slide image
        channels = imageReader.get_channels(img_path)
        logger.info(f"Channels found: {channels}")
        percentiles, backgrounds = norm.compute_slide_normalization(img_path, channels)
        logger.info(f"Computed percentiles: {percentiles}, backgrounds: {backgrounds}")
        if multi_thread == True:
            # Parallel processing of volume tiles to pcd
            merged_pcd = tiling.process_tiles_multi_threaded(img_path, percentiles, backgrounds, channels, block_size = block_size, max_workers = num_workers)
        else:
            # Sequential processing of volume tiles to pcd
            merged_pcd = tiling.process_tiles_sequential(img_path, percentiles, backgrounds, channels, block_size = block_size)

        # Save global point cloud in the requested format, which defaults to the input format
        formats = _output_formats(output_format, is_ome_zarr)

        if 'zarr' in formats:
            # Write into an OME-Zarr structure under points/adaptive_sampling_3D
            zarr_root = os.path.join(result_dir, f"{base_name}.ome.zarr")
            logger.info(f"Saving point cloud to OME-Zarr: {zarr_root}/{pointCloudWriter.POINTS_GROUP}/{pointCloudWriter.POINTS_NAME}")
            pointCloudWriter.save_pcd_ome_zarr(merged_pcd, zarr_root)

        if 'ply' in formats:
            result_path = os.path.join(result_dir, f"{base_name}.ply")
            logger.info(f"Saving point cloud to: {result_path}")
            pointCloudWriter.save_pcd(merged_pcd, result_path)
                    
if __name__ == "__main__":
    main()



