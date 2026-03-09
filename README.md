# Adaptive sampling for 3D biopsies

This project provides a scalable pipeline for converting large 3D microscopy volumes stored in HDF5 into light-weight colored point clouds using tile-based processing and adaptive sampling. The system is designed for datasets that exceed RAM capacity and supports parallel processing, automatic normalization, and adaptive density/information sampling.

## Key Features

* **Tiling**: Large volumes are processed by chunks (`block_size`) using h5py "lazy loading".
* **Multi-threading**: Tiles are processed in individual threads and resulting point cloud is merged in the main thread.
* **Color Mapping**: Maps original voxel intensities from volume channels back onto the generated point cloud.
* **Auto-Normalization**: Automatic estimation and removal of global background and percentile normalization before processing.
* **Adaptive Density Sampling**: Sigmoid function determines tile density (n points to sample per tile) based on mesh surface area.
* **Adaptive Information Sampling**: Informaiton-based sampling based on Sobel gradient, mesh surface, mesh curvature (to be tested), and intensity maps (to be implemented).

## Installation

### 1) Clone repository

git clone https://github.com/alpnboy/adaptive_sampling.git  
cd adaptive_sampling

### 2) Create new environment and install requirements

conda create -n myenv python=3.10  
conda activate myenv  
pip install -r requirements.txt

## Pipeline overview

H5 Volume  
↓  
Whole slide estimates (intensities cutoffs, percentiles, ...)  
↓  
Tile-based Processing (normalization, meshing, ...)  
↓  
Adaptive Sampling  
↓  
Tile Colored Point Cloud  
↓  
Merged Global Point Cloud  
↓  
PLY Export

## Project structure

adaptive_sampling/              # Root  
├── data/                       
│   ├── raw_h5/                 # Input: original .h5 image volume  
│   └── results/                # Output: colored point cloud in .ply format  
├── adaptive_sampling/            
│   ├── __init__.py  
│   ├── core/                   # Logic for Mesh, PointCloud, and Volume objects  
│   │   ├── __init__.py  
│   │   ├── mesh.py  
│   │   ├── pointcloud.py  
│   │   └── volume.py  
│   ├── io/                     # Input (H5) and Output (PLY) operations  
│   │   ├── __init__.py  
│   │   ├── read.py  
│   │   └── write.py  
│   └── stages/                 # Processing stages  
│       ├── __init__.py  
│       ├── normalization.py  
│       └── tile_processing.py  
└── create_pcd_dataset.py       # Main entry point

## Usage guide

The `create_pcd_dataset.py` file is the main entry point for the pipeline. It orchestrates the reading, pre-processing, processing, and writing of the output color point clouds from H5 volume images.

### Execution Example

python adaptive_sampling.create_pcd_dataset.py --input_dir ./data/raw_h5 --result_dir ./data/results --multi_thread True --num_workers 8 --block_size 500 500 500

## Example Output

The pipeline produces colored point clouds in `.ply` format. These can be visualized using:

- MeshLab  
- Open3D  
- CloudCompare  

Point cloud can appear flat without normals and rendering. For best visualization open the .ply file with CloudCompare, increase point size to 2-3, and enable "SSAO Shader" under "Display/Shaders & Filters".