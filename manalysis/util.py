import numpy as np
import time
import re

__all__ = ['longest_sequence',
           'longest_cont_segment',
           'get_percentile_limits',
           'natural_sort',
           'chunkify',
           'is_notebook',
           'get_TFS_metadata',
          ]

def longest_sequence(l):
    """Get longest island of contineous sequence of 1s

    Parameters
    ----------
    l : list or np.array
        

    Examples
    --------
    >>> l = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0 ,1]
    >>> longest_sequence(l)
    (5, 9)

    References
    ----------
    [1] https://stackoverflow.com/a/38161867
    """
    l = np.array(l)
    l = np.diff(np.hstack(([False],l==1,[False])))
    # Get start, stop index pairs for islands/seq. of 1s
    idx_pairs = np.where(l)[0].reshape(-1,2)

    # Get the island lengths, whose argmax would give us the ID of 
    # longest island. Start index of that island would be the 
    # desired output
    start, stop = idx_pairs[np.diff(idx_pairs, axis=1).argmax()]

    return start, stop


def longest_cont_segment(y, discon=20):
    """Get longest conineous segment

    Parameters
    ----------
    y : np.array 1D
        Serie of data (i.e. vibration amplitude)
    discon : int or float
        Value determining when series is discontineous
    
    Returns
    -------
    start : int
        Start index
    stop : int
        Stop index
    """
    # Differentiate
    d = np.diff(y, prepend=y[0])
    abs_d = np.abs(d)

    # Indices where y < discon
    indices = np.argwhere(abs_d < discon).ravel()
    # Separate into islands of contineous 1s
    islands = np.diff(indices)
    # Get longest island
    start, stop = longest_sequence(islands)

    start = indices[start]
    stop = indices[stop]
    return start, stop


def get_percentile_limits(n, bins, percentile=0.99):
    total = np.sum(n)
    fraction = (1.0 - percentile) * total / 2
    # First right side index where cumsum(counts) exeeds percintile limit
    idx1 = np.argwhere(np.cumsum(n) > fraction)[0][0]
    # First left side index where cumsum(counts) exeeds percintile limit
    idx2 = np.argwhere(np.cumsum(n[::-1]) > fraction)[0][0]
    
    lim1 = bins[idx1]
    lim2 = bins[::-1][idx2]
    return lim1, lim2


def natural_sort(l):
    """A more natural sorting algorithm

    Parameters
    ----------
    l : list
        List of strings in need of sorting

    Examples
    --------
    >>> l = ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']
    >>> sorted(l)
    ['Elm11', 'Elm12', 'Elm2', 'elm0', 'elm1', 'elm10', 'elm13', 'elm9']
    >>> natural_sort(l)
    ['elm0', 'elm1', 'Elm2', 'elm9', 'elm10', 'Elm11', 'Elm12', 'elm13']

    References
    ----------
    [1] https://stackoverflow.com/a/4836734/5285918
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def chunkify(l, n, duplicate_edges=False):
    """Divide list into (roughly) equal chunks,
    possibly duplicating inner edges

    Parameters
    ----------
    l : list
        list to chunkify

    Examples
    --------


    References
    ----------
    [1] Adapted from https://stackoverflow.com/a/2135920
    """
    k, m = divmod(len(l), n)
    new_l = list(l[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    if duplicate_edges:
        for j in range(1, len(new_l)):
            new_l[j].insert(0, new_l[j-1][-1])
    return new_l


def is_notebook():
    """Attempts to determines whether code is being exectued in a notebook or not
    
    References
    ----------
    [1] https://stackoverflow.com/a/39662359/5285918
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
  
   
def get_TFS_metadata(fn, keys):
    """
    Finds key:value pairs in Thermo Fischer Scientific metadata based 
    on keys provided. 
    
    Parameters
    ----------
    fn : Path or str
        Path of the image containg TFS metadata
    keys : str or list of str
        Keys to look for in metadata

    Returns
    -------
    metadata : dictionary
        Dictionary containing key:value pairs. Attempts to return 
        floats as value, fallback is str.
        
    or
    
    metadata : str or float
        If a single key is provided. When none of the keys
        are found returns empty string.
    
    """
    if type(keys) == str: 
        keys = [keys]
    metadata = {}
    f = open(fn, 'r', errors='ignore')
    lines = f.readlines()
    for line in lines[::-1]:
        if any(key+"=" in line for key in keys):
            key, val = line.split("=")
            try:
                metadata[key] = float(val.strip())
            except:
                metadata[key] = val.strip()
            if len(metadata.keys()) == len(keys): 
                break
    if len(metadata) < 1:
        return ""
    elif len(metadata) < 2:
        return list(metadata.values())[0]
    else:
        return metadata
        

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))