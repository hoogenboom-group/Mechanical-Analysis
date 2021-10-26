import numpy as np
import logging

from .math import fast_corr
from .util import get_TFS_metadata
from .filter import hp_filter_vibrations

__all__ = ['generate_heavisides',
           'extract_shifts',
           'extract_vibrations',
          ]


def generate_heavisides(N, y0=0.5):
    # Create NxN upper triangular matrix filled with 1s
    a = (1 - np.tri(N,N))
    # Fill diagonal with half value (default 0.5)
    np.fill_diagonal(a, y0)
    return a

    
def extract_shifts(data):
    """Extract line-to-line shifts from scanning type microscopy
    data. The image Y axis is the slow scan direction (time)
    and line-to-line shifts along the image X direction are
    extracted. 

    Parameters
    ----------
    data : 2D numpy array
        Scanning image data to be processed

    Returns
    -------
    shifts : 1D numpy array of length data.shape[0]
        Line-to-line shifts in pixels

    """
    # Generate array with Heaviside step at different location
    hvs = generate_heavisides(data.shape[-1])
    hvs *= np.mean(data)
    # Vectorized Pearson correlation 
    pcc = fast_corr(np.rot90(data), np.rot90(hvs))
    # Get edge locations, look for maximum in both
    # correlation and anti correlation coefficient
    shifts = np.argmax(np.abs(pcc), axis=1)
    return shifts
    

def extract_vibrations(data, file_path=None, pixel_width=None, 
                       line_time=None):
    """Extract and convert shifts to physical quantities, 
    i.e. shifts in [nm] versus time.

    Parameters
    ----------
    data : 2D numpy array
        Scanning image data to be processed
    file_path : path or str
        Path of the image to get metadata from
    pixel_width : float
        Pixel size of image in [nm]
    line_time : float
        Sampling time in [s]

    Returns
    -------
    x : 1D numpy array of length data.shape[0]
        Time [s]
    y : 1D numpy array of length data.shape[0]
        Vibration amplitude [nm]
    

    """
    # If user does not supply anything
    if not (pixel_width and line_time):
        if not file_path:
            raise TypeError("Must provide file path if not providing"
                            "pixel_width and line_time.")
        # Try extracting TFS metadata
        try:
            logging.info("Trying to get TFS metadata.")
            meta = get_TFS_metadata(file_path, ["PixelWidth", "LineTime"])
        except: 
            logging.info("Failed to get TFS metadata.")
        else: 
            pixel_width = meta["PixelWidth"]
            line_time = meta["LineTime"]
        # Try others?

    # Extract shifts
    shifts = extract_shifts(data)
    # Vibration amplitude in [nm]
    y = np.array(shifts, dtype=float) * pixel_width * 1e9
    # Number of samplepoints
    N = len(y)
    # Time series in [s]
    x = np.linspace(0.0, N*line_time, N) 
    return x, y