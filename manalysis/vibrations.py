from pathlib import Path

import numpy as np
import pandas as pd
import logging
import scipy

from .math import fast_corr
from .util import get_TFS_metadata, longest_cont_segment
from .filter import hp_filter_vibrations, scalloping_loss_corrected_fft
from .io import get_images

__all__ = ['generate_heavisides',
           'extract_shifts',
           'extract_vibrations',
           'batch_extract',
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
                       line_time=None, direction=None):
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
            meta = get_TFS_metadata(file_path, ["PixelWidth", "LineTime", "ScanRotation"])
        except: 
            logging.info("Failed to get TFS metadata.")
        else: 
            pixel_width = meta["PixelWidth"]
            line_time = meta["LineTime"]
            if meta["ScanRotation"] == 0.0:
                direction = 'x'
            elif 1.555 < meta["ScanRotation"] < 1.586:
                direction = 'y'
            else:
                raise TypeError("ScanRotation unknown.")
        # Try others?

    # Extract shifts
    shifts = extract_shifts(data)
    # Vibration amplitude in [nm]
    y = np.array(shifts, dtype=float) * pixel_width * 1e9
    # Number of samplepoints
    N = len(y)
    # Time series in [s]
    x = np.linspace(0.0, N*line_time, N) 
    return direction, x, y

    
def batch_extract(dir_path, load_new=False, image_fraction=0.33,
                  smooth=0):
    """Extract and convert shifts to physical quantities, 
    i.e. shifts in [nm] versus time for whole directory 
    of images.

    Parameters
    ----------
    dir_path : path or str
        Path of the directory to process

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing processed data from images
        inside directory.

    """
    names = ["Raw time [s]", "Raw displacement [nm]", 
             "HPF(raw displacement) [nm]", "Time [s]", 
             "Displacement [nm]", "Frequency [Hz]", 
             "P2P amplitude [nm]"]
    dfs, avgs = {}, {}
    csv_location = Path(dir_path) / "Vibration_data.csv"
    if csv_location.exists() and not load_new:
        df = pd.read_csv(csv_location, header=[0,1,2])
        return df
    
    imgs = get_images(dir_path)
    for fp, img in imgs:
        if not smooth == 0: img = scipy.ndimage.gaussian_filter1d(img, smooth, 1)
        direction, x, y = extract_vibrations(img, fp)

        start, stop = longest_cont_segment(y)       
        y_sel = y[start:stop]
        x_sel = x[:stop-start]
        if stop-start < image_fraction*len(y): 
            continue

        y_hpf = hp_filter_vibrations(x_sel, y_sel)
        xf, yf = scalloping_loss_corrected_fft(y_hpf, x_sel[1]-x_sel[0])
        avgs.setdefault(direction, []).append(pd.Series(yf, index=xf))
        data = [x, y, y_hpf, x_sel, y_sel, xf, yf]
        d = {name:val for name, val in zip(names, data)}
        dfs[(direction, fp)] = pd.DataFrame.from_dict(d, orient='index').transpose()

    for key in avgs.keys():
        avg = pd.concat(avgs[key], axis=1).interpolate('index').mean(axis=1)
        median = pd.concat(avgs[key], axis=1).interpolate('index').median(axis=1)
        
        data = [avg.index.values, avg.values]
        d = {name:val for name, val in zip(names[-2:], data)}
        dfs[key, 'Average'] = pd.DataFrame.from_dict(d, orient='index').transpose()
        
        data = [median.index.values, median.values]
        d = {name:val for name, val in zip(names[-2:], data)}
        dfs[key, 'Median'] = pd.DataFrame.from_dict(d, orient='index').transpose()

    df = pd.concat(dfs, axis=1, keys=dfs.keys())
    df.to_csv(csv_location, index=False)
    return df
    
    
    



    
    