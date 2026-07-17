import numpy as np

from adaptive_sampling_3D.core import volume as vlm
from adaptive_sampling_3D.io import imageReader


def compute_slide_normalization(img_path, channels):
    """ Estimates 

    Args:
        img_path (str): Path to the HDF5 file containing the image volume.
        channels (list): Indexes of the channels to estimate percentiles and background cutoffs.

    Returns:
        tuple: A tuple of of tuples of channel-wise percentiles (low_p, high_p) and background cutoffs (int).
    """
    
    histograms = []
    
    # Estimate intensity histograms over image volume
    for channel in channels:
        hist = imageReader.sample_volume_histograms(img_path, channel)
        histograms.append(hist) 
            
    hist_ch1 = histograms[0].astype(np.float32)
    hist_ch2 = histograms[1].astype(np.float32)
        
    # Estimate background cutoffs over all slide
    background_ch1, _ = vlm.volume_background_cutoff(hist_ch1)
    background_ch2, _ = vlm.volume_background_cutoff(hist_ch2)
    backgrounds = (background_ch1, background_ch2)
        
    # Background subtraction on all slide histograms
    hist_ch1 = vlm.volume_background_subtraction(hist_ch1, background_ch1)
    hist_ch2 = vlm.volume_background_subtraction(hist_ch2, background_ch1)
        
    # Min/Max percentiles of background adjusted all slide
    percentile_ch1 = vlm.volume_percentiles(hist_ch1)          # (low_p, high_p)
    percentile_ch2 = vlm.volume_percentiles(hist_ch2)
    percentiles = (percentile_ch1, percentile_ch2)
    
    return percentiles, backgrounds
        
        