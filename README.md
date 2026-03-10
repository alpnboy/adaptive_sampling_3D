# Adaptive sampling for 3D biopsies

This project provides a scalable pipeline for converting large 3D microscopy volumes stored in HDF5 into light-weight colored point clouds using tile-based processing and adaptive sampling. The system is designed for datasets that exceed RAM capacity and supports parallel processing, automatic normalization, and adaptive density/information sampling.

## Key Features

* **Tiling**: Large volumes are processed by chunks (`block_size`) using h5py "lazy loading".
* **Multi-threading**: Tiles are processed in individual threads and results are merged in the main one.
* **Auto-Normalization**: Automatic background subtraction and percentile normalization before processing.
* **Adaptive Density Sampling**: Sigmoid function determines tile density (n points to sample per tile) based on mesh surface area.
* **Adaptive Information Sampling**: Informaiton-based sampling based on Sobel gradient, mesh surface, mesh curvature (to be tested), and intensity maps (to be implemented).

## Installation

### 1) Clone repository
```bash
git clone https://github.com/alpnboy/adaptive_sampling_3D.git  
cd adaptive_sampling
```

### 2) Create new environment and install requirements

```bash
conda create -n myenv python=3.10  
conda activate myenv  
pip install -r requirements.txt
```

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

## Usage guide

The `create_pcd_dataset.py` file is the main entry point for the pipeline. It orchestrates the reading, processing, and writing of colored point clouds from H5 images.

### Command Example
```bash
python adaptive_sampling.create_pcd_dataset.py --input_dir ./data/raw_h5 --result_dir ./data/results --multi_thread True --num_workers 8 --block_size 500 500 500
```
## Example Output

The pipeline produces colored point clouds in `.ply` format. These can be visualized using:

- MeshLab  
- Open3D  
- CloudCompare

Point clouds can appear flat without normals and rendering. For best visualization open the .ply file with CloudCompare, increase point size to 2-3, and enable "SSAO Shader" under "Display/Shaders & Filters".

![Example output](images/example_1.png)

## Requirements

### Input HDF5 Structure

At present, the pipeline expects each `.h5` file to contain **two datasets**, each representing a volumetric image channel.

Example:

```text
file.h5
├── channel_0
├── channel_1
```

### Packages

The project requires Python ≥ 3.10 and the following main libraries:

- numpy
- h5py
- open3d
- scikit-image
- scipy
- tqdm

All dependencies are listed in `requirements.txt`.
