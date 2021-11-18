import logging

from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from . import drift
from .io import get_images
from .util import is_notebook, chunkify

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


__all__ = ['parallel_process_directory',
           ]


def parallel_process_directory(dir_path, load_new=False,
                               num_cores=4):
    """Parallelized version of `process_directory' 

    Parameters
    ----------
    ...
    num_cores: int
        Number of cores to use

    Returns
    -------
    ...
        
    Notes
    -----
    ...
    """
    if Path(dir_path).is_dir():
        csv_location = Path(dir_path) / "Drift_data.csv"
        if csv_location.exists() and not load_new:
            df = pd.read_csv(csv_location)
            return df
        
        imgs = get_images(dir_path)
        imgs = chunkify(imgs, num_cores, duplicate_edges=True)
        dfs = Parallel(n_jobs=num_cores) \
                       (delayed(drift.batch_extract) \
                       (imgs[i]) \
                       for i in range(len(imgs)))
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(csv_location, index=False)
        return df
    