from pathlib import Path
from skimage.registration import phase_cross_correlation

import numpy as np
import pandas as pd
import logging
import scipy

from .util import get_TFS_metadata
from .io import get_images

__all__ = ['extract_shift',
           'extract_drift',
           'batch_extract',
          ]


def extract_shift(im1, im2, upsample=10, method=phase_cross_correlation):
    """Extract image-to-image shift from scanning type microscopy
    data. 
    
    Parameters
    ----------
    im1 : 2D numpy array
        Scanning image data to be processed

    im2 : 2D numpy array
        Scanning image data to be processed

    Returns
    -------
    shift : 2-valued tuple 
        y, x shift in pixels

    """
    shift, error, diffphase = method(im1, im2, upsample_factor=upsample)
    return shift


def extract_drift(im1, im2, pixel_width=None, dt=None):
    # If user does not supply anything
    transpose = False
    if not (pixel_width and dt):
        fp1, im1 = im1
        fp2, im2 = im2
        # Try extracting TFS metadata
        try:
            logging.info("Trying to get TFS metadata.")
            meta = get_TFS_metadata(fp1, ["PixelWidth", "ScanRotation"])
        except: 
            logging.info("Failed to get TFS metadata.")
        else: 
            pixel_width = meta["PixelWidth"]
            if meta["ScanRotation"] == 0.0:
                pass
            elif 1.555 < meta["ScanRotation"] < 1.586:
                transpose = True
            else:
                raise TypeError("ScanRotation unknown.")
        dt = Path(fp2).stat().st_mtime - Path(fp1).stat().st_mtime
        # Try others?

    # Extract shifts
    shift = extract_shift(im1, im2)
    # Drift in [nm]
    shift = np.array(shift, dtype=float) * pixel_width * 1e9
    if transpose: shift = shift[::-1]
    return dt, shift[0], shift[1]


def batch_extract(imgs):
    """
    
    """
    names = ["dt [s]", "y shift [nm]", "x shift [nm]", "fp1", "fp2"]
    df_list = []
    for i in range(1, len(imgs)):
        dt, y, x = extract_drift(imgs[i-1], imgs[i])
        df_list.append([dt, y, x, imgs[i-1][0], imgs[i][0]])
    df = pd.DataFrame(df_list, columns=names)
    return df


def process_directory(dir_path, load_new=False):
    """Extract and convert image to image shift to physical 
    quantities, i.e. drift in [nm] versus time for whole 
    directory of images.

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
    if Path(dir_path).is_dir():
        csv_location = Path(dir_path) / "Drift_data.csv"
        if csv_location.exists() and not load_new:
            df = pd.read_csv(csv_location)
            return df
        
        imgs = get_images(dir_path)
        df = batch_extract(imgs)
        df.to_csv(csv_location, index=False)
        return df
    