# Adaptive sampling for 3D biopsies

This project provides a scalable pipeline for converting large 3D microscopy volumes stored in HDF5 into light-weight colored point clouds using tile-based processing and adaptive sampling. The system is designed for datasets that exceed RAM capacity and supports parallel processing, automatic normalization, and adaptive density/information sampling. The final colored point cloud is meant to be a highly compressed representation compatible with existing end-to-end algorithms for geometric deep learning.

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
conda create -n 3DxNenv python=3.10  
conda activate 3DxNenv  
pip install -r requirements.txt
```

Python venvs can be used instead of conda if the host system has a suitable Python version (>=3.10).
```bash
# On Linux/OSX:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline overview
```text
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
```

## Project structure

```text
adaptive_sampling_3D/              # Root
├── data/                       
│   ├── raw_h5/                 # Input H5 volumes
│   └── results/                # Output colored point clouds
├── adaptive_sampling_3D/          
│   ├── core/                   # Mesh, PointCloud, Volume logic
│   ├── io/                     # Input/Output operations
│   └── stages/                 # Processing stages
└── create_pcd_dataset.py       # Main entry point
```


## Usage guide

The `create_pcd_dataset.py` file is the main entry point for the pipeline. It orchestrates the reading, processing, and writing of colored point clouds from H5 images.

### Command Example
```bash
python -m adaptive_sampling_3D.create_pcd_dataset --input_dir ./data/raw_h5 --result_dir ./data/results --multi_thread True --num_workers 8 --block_size 500 500 500
```

### Configuration / Parameters
```text
- `--input_dir`: directory with input volumes (HDF5 `.h5` or OME-Zarr `.zarr`)
- `--result_dir`: output directory for results
- `--output_format`: `auto` (default, matches the input), `ply`, `zarr`, or `both`
- `--block_size`: size of tile blocks in z, y, x
- `--multi_thread`: True/False
- `--num_workers`: number of threads
```
## Example Output

The pipeline produces colored point clouds in `.ply` format. These can be visualized using:

- MeshLab  
- Open3D  
- CloudCompare

Point clouds can appear flat without normals and rendering. For best visualization open the .ply file with CloudCompare, increase point size to 2-3, and enable "SSAO Shader" under "Display/Shaders & Filters".

![Example output](images/example_1.png)

## Requirements

### Input structure

The pipeline reads two input formats, auto-detected from each entry in `--input_dir`:

**HDF5** (`.h5`) — each file must contain **two datasets** of shape (z, y, x), each a volumetric image channel:

```text
file.h5
├── channel_0
├── channel_1
```

**OME-Zarr v0.5** (`.zarr` / `.ome.zarr`) — a Zarr v3 group read via `ome-zarr-py`. Channels live on the `c` axis of the multiscale array (axes such as `czyx` or `tczyx`); the pipeline uses the highest-resolution level and its first two channels.

Either on-disk layout is accepted. `--input_dir` may hold multiscale image groups directly, or `bioformats2raw` containers whose root carries `"bioformats2raw.layout": 3` and nests the image in a numbered subgroup:

```text
sample.ome.zarr/
├── zarr.json           # "bioformats2raw.layout": 3
├── OME/
│   ├── zarr.json       # "series": ["0"]
│   └── METADATA.ome.xml
└── 0/                  # the multiscale image
```

Containers are resolved to their image group automatically, preferring the `series` attribute and otherwise probing the consecutively numbered groups the spec mandates. A container holding several series reads the first and logs a warning; pass the image group path directly to select another.

### Output structure

The point cloud carries no trace of the container it was read from, so `--output_format` selects the output independently of the input:

| `--output_format` | Result |
|---|---|
| `auto` *(default)* | Follows the input: PLY for HDF5, OME-Zarr for OME-Zarr |
| `ply` | `<result_dir>/<name>.ply` for any input |
| `zarr` | `<result_dir>/<name>.ome.zarr` for any input |
| `both` | Writes both of the above |

- PLY output → a colored point cloud in `.ply` format in `--result_dir`.
- OME-Zarr output → the point cloud is written into an OME-Zarr structure at `<result_dir>/<name>.ome.zarr/points/adaptive_sampling_3D`, as a Zarr v3 sub-group holding `coordinates` (N×3, z/y/x) and `colors` (N×3, RGB) arrays.

Note that PLY has no field for axis order, so the `(z, y, x)` coordinates the OME-Zarr writer records explicitly under `axes` land unlabelled in PLY's `x/y/z`. PLY also quantises colors to 8 bits, so a `both` run's two outputs agree on coordinates exactly but differ by up to 1/255 in color.

### Packages

The project requires Python ≥ 3.10 and the following main libraries:

- numpy
- h5py
- open3d
- scikit-image
- scipy
- tqdm
- ome-zarr
- zarr

All dependencies are listed in `requirements.txt`.

## Future developments

### User:
- Integrate information-based sampling using local intensities map and surface curvature.
- Enable user specified cloud density and target channel/s.
- Expand reader to OME-ZAR images.

### Performance:
- Optimize cpu operations (meshing, gradient estimation, ...).
- C++ parallel processing implementation with Python bindings.
