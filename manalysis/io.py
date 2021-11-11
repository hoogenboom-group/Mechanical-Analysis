from pathlib import Path
import logging

import numpy as np
import pandas as pd

from skimage import io
from skimage import img_as_ubyte

from .util import is_notebook, natural_sort,\
                  get_TFS_metadata

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = ['get_images',
           ]


def get_images(file_pattern):
    """Get all images to be processed

    Parameters
    ----------
    file_pattern : list or str
        A directory from which all images will be processed

    Returns
    -------
    images : list of tuples
        List of tuples of the form (filepath, image)

    Examples
    --------
    * `file_pattern` is a directory
    >>> file_pattern = '/path/to/data/'
    >>> get_images(file_pattern)

    """

    
    # If a directory or individual filename
    if isinstance(file_pattern, str):
        # Directory
        if Path(file_pattern).is_dir():
            logging.info("Creating list from directory.")
            # Collect every png/tif/tiff image in directory
            filepaths = list(Path(file_pattern).glob('**/*.png')) + \
                        list(Path(file_pattern).glob('**/*.tif')) + \
                        list(Path(file_pattern).glob('**/*.tiff'))
            # Sort filepaths
            filepaths = natural_sort([fp.as_posix() for fp in filepaths])
            # Load images
            images = []
            for i, fp in enumerate(filepaths):
                image = img_as_ubyte(io.imread(fp))
                logging.info(f"Reading image file ({i+1}/{len(filepaths)}) : {fp}")
                # Omit TFS databar when present
                y_size = get_TFS_metadata(fp, "ResolutionY")
                if y_size:
                    if image.shape[0] > y_size:
                        image = image[:int(y_size)]
                images.append((fp, image))

        # ?
        else:
            if Path(file_pattern).exists():
                raise ValueError(f"Not sure what to do with `{file_pattern}`.")
            else:
                raise ValueError(f"`{file_pattern}` cannot be located or "
                                  "does not exist.")

    else:
        raise TypeError("Must provide a directory, "
                        f"not {type(file_pattern)}.")

    # Return list
    logging.info(f"{len(images)} images loaded created succesfully.")
    return images