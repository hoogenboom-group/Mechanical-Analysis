import matplotlib.transforms as mtransforms
import matplotlib.scale as mscale
import matplotlib.ticker as ticker


import numpy as np
import re

__all__ = ['longest_sequence',
           'longest_cont_segment',
           'natural_sort',
           'is_notebook',
           'register_squarerootscale',
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


def longest_cont_segment(y, discon=10):
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


def register_squarerootscale():
    """Registers squareroot scale for matplotlib
    
    References
    ----------
    [1] https://stackoverflow.com/a/39662359/5285918
    """
    mscale.register_scale(SquareRootScale)
   
   
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

class SquareRootScale(mscale.ScaleBase):
    """
    ScaleBase class for generating square root scale.
    
    References
    ----------
    [1] https://stackoverflow.com/questions/42277989/square-root-scale-using-matplotlib-python
    """
 
    name = 'squareroot'
 
    def __init__(self, axis, **kwargs):
        # note in older versions of matplotlib (<3.1), this worked fine.
        # mscale.ScaleBase.__init__(self)

        # In newer versions (>=3.1), you also need to pass in `axis` as an arg
        mscale.ScaleBase.__init__(self, axis)
 
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(ticker.AutoLocator())
        axis.set_major_formatter(ticker.ScalarFormatter())
        axis.set_minor_locator(ticker.NullLocator())
        axis.set_minor_formatter(ticker.NullFormatter())
 
    def limit_range_for_scale(self, vmin, vmax, minpos):
        return  max(0., vmin), vmax
 
    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True
 
        def transform_non_affine(self, a): 
            return np.array(a)**0.5
 
        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()
 
    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True
 
        def transform(self, a):
            return np.array(a)**2
 
        def inverted(self):
            return SquareRootScale.SquareRootTransform()
 
    def get_transform(self):
        return self.SquareRootTransform()