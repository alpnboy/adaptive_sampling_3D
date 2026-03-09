import numpy as np
import open3d as o3d
import h5py as h5

from scipy.stats import shapiro, skew
from skimage import measure
from scipy import ndimage



def compute_sobel_gradient(volume):
    """Compute the Sobel gradient combined magnitudes (Z, Y, X) describing the edges of a 3D image volume.

    Args:
        volume (np.ndarray): A 3D voxel grid of shape (Z, Y, X) containing image intensities.

    Returns:
        np.ndarray: Combined magnitude of the Sobel gradient with shape (Z, Y, X).
    """
    
    # Compute image gradient for three axis
    grad_x = ndimage.sobel(volume, axis=0)
    grad_y = ndimage.sobel(volume, axis=1)
    grad_z = ndimage.sobel(volume, axis=2)
    
    # Combined magnitude of the gradients
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2) 
    
    return gradient_magnitude


def sample_volume_histograms(img_path, channel, n_samples = 10000):
    """ Estimate the intensitiy distribution of a 3D image volume by sampling.

    Args:
        img_path (str): Path to the HDF5 file containing the image volume.
        channel (int): Index of the channel to sample.
        n_samples (int, optional): Number of random voxels to sample for the distribution estimate. Defaults to 10000.

    Returns:
        np.ndarray: A 1D array of length `n_samples` containing the sampled intensity values.
    """
            
    # Open volume HD5F image        
    with h5.File(img_path, "r") as f:
            
        # Get dataset shape to know volume voxel boundaries    
        dataset = f[channel]        
        vol_size = dataset.shape
            
        # Random sample n x (z, y, x) coordinates within volume boundaries
        random_coordinates = np.random.randint([0, 0, 0], vol_size, size = (n_samples, 3))
        sampled_values = np.empty(n_samples, dtype=dataset.dtype)
        
        # Lazy load chunk coordinates from HD5F image    
        for i, (z, y, x) in enumerate(random_coordinates):
            sampled_values[i] = dataset[z, y, x]
                
        return sampled_values


def volume_background_cutoff(hist, cutoff_range = (20, 200), step = 5, max_samples = 5000, min_voxels = 200, random_state = 2):
    """ Given a distribution of intensities finds background cutoff by iteratively fitting a normal distribution on increasing intensities interval.
    Background noise is assumed to be normally distributed within left tail of intensities distribution.
    

    Args:
        hist (np.ndarray): Intensities sampled from image volume.
        cutoff_range (tuple, optional): Lower and upper bound defining an interval to iteratively search the background cutoff. Defaults to (20, 200).
        step (int, optional): _description_. Defaults to 5.
        max_samples (int, optional): Max number of background samples allowed to be fitted to normal distribution. Defaults to 5000.
        min_pixels (int, optional): Minimum number of voxels within estimated background. Defaults to 200.
        random_state (int, optional): Seed for np.random. Defaults to 2.

    Returns:
        tuple: A tuple with best cutoff (int) and fitting scores (set).
    """
    
    rng = np.random.default_rng(random_state)
    hist = np.asarray(hist, dtype=np.float32)
    hist = hist[np.isfinite(hist)]

    cutoffs = np.arange(
        cutoff_range[0],
        cutoff_range[1] + step,
        step
    )

    scores = {}

    for cutoff in cutoffs:
        bg_pixels = hist[hist <= cutoff]

        if len(bg_pixels) < min_voxels:
            continue

        if len(bg_pixels) > max_samples:
            bg_pixels = rng.choice(bg_pixels, max_samples, replace=False)

        bg_pixels = bg_pixels - cutoff

        if np.std(bg_pixels) < 1e-6:
            continue

        try:
            W, _ = shapiro(bg_pixels)
            scores[cutoff] = W
            
        except Exception:
            scores[cutoff] = -abs(skew(bg_pixels))

    if len(scores) == 0:
        print("No valid cutoff in given range")
        return 0
    
    # Best fit cutoff for the background
    background_cutoff = max(scores, key=scores.get)
    
    return background_cutoff, scores


def volume_background_subtraction(volume, background_cutoff):
    """Subtracts background level and clips negative values to zero.

    Args:
        volume (np.ndarray): Input image volume (typically float32).
        background_level (float): Intensity value to subtract.

    Returns:
        np.ndarray: Background-subtracted volume.
    """
    # Background subtraction and clip
    volume = volume - background_cutoff
    volume[volume < 0] = 0
        
    return volume


def volume_percentiles(values):
    """Calculates the 1st and 99th percentiles of the input values.

    Args:
        values (np.ndarray): Array of intensity samples.

    Returns:
        tuple: (low_p, high_p) as floats.
    """
    
    low_p  = np.percentile(values, 1)
    high_p = np.percentile(values, 99)
    
    return low_p, high_p


def volume_normalization(volume, percentile_levels):
    """Normalize volume to [0, 1] based on percentiles levels.

    Args:
        volume (np.ndarray): Input volume to normalize.
        percentile_levels (tuple): (1st, 99th) percentiles cutoffs.

    Returns:
        np.ndarray: Percentiles normalized volume clipped between 0 and 1.
    """
    
    # prevent 0 divison
    eps = 1e-9
    
    # Normalize
    low_p, high_p = percentile_levels
    normalized_volume = (volume - low_p) / (high_p - low_p + eps)
    
    # Clip between 0 and 1
    normalized_volume[normalized_volume > 1] = 1
    normalized_volume[normalized_volume < 0] = 0
    
    return normalized_volume


def volume_to_mesh_marching_cubes(volume, threshold = 0.4):
    """Generates mesh from volume applying marching cube algorithm on voxel grid.

    Args:
        volume (np.ndarray): A 3D voxel grid of shape (Z, Y, X) containing image intensities.
        threshold (float, optional): Binary cutoff for mesh generation.

    Returns:
        open3d.geometry.TriangleMesh: Triangle mesh describing volume isosurface.
    """
    
    # Estimate and normalize gradient magnitude
    gradient = compute_sobel_gradient(volume)
    low, high = volume_percentiles(gradient)
    gradient = volume_normalization(gradient, [low, high])
    
    # Generate combined volume scalar using original image volume and its gradient
    stacked = np.stack([gradient, volume], axis = 3)
    weights = [0.5,0.5]
    volume_scalar = np.tensordot(stacked, weights, axes=([3],[0]))
    
    # Perform marching cubes at given levelset threshold
    if np.max(volume_scalar) < threshold:
        return []
    
    try:
        verts, faces, _, _ = measure.marching_cubes(volume_scalar, level = threshold)
    except:
        return []
    
    # Add vertices and triangles to mesh object
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return mesh