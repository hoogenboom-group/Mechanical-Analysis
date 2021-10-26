import numpy as np


__all__ = ['calculate_non_mask_overlaps',
           'fast_cov',
           'validate_inputs',
           'nan_fast_cov',
           'fast_corr',
          ]


def _fast_dot_divide(x, y, destination):
    """helper method for use within the _fast_cov method - carry out the dot product and subsequent 
    division to generate the covariance values.  For use when there are no missing values.
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy
    """
    np.dot(x.T, y, out=destination)
    np.divide(destination, (x.shape[0] - 1), out=destination)


def calculate_non_mask_overlaps(x_mask, y_mask):
    """for two mask arrays (x_mask, y_mask - boolean arrays) determine the number of entries in common there would be for each 
    entry if their dot product were taken
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy    
    """
    x_is_not_nan = 1 * ~x_mask
    y_is_not_nan = 1 * ~y_mask

    r = np.dot(x_is_not_nan.T, y_is_not_nan)
    return r


def _nan_dot_divide(x, y, destination):
    """helper method for use within the _fast_cov method - carry out the dot product and subsequent
    division to generate the covariance values.  For use when there are missing values.
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy
    """
    np.ma.dot(x.T, y, out=destination)

    divisor = calculate_non_mask_overlaps(x.mask, y.mask) - 1

    np.ma.divide(destination, divisor, out=destination)


def fast_cov(x, y=None, destination=None):
    """calculate the covariance matrix for the columns of x (MxN), or optionally, the covariance matrix between the
    columns of x and and the columns of y (MxP).  (In the language of statistics, the columns are variables, the rows
    are observations).

    Args:
        x (np array-like) MxN in shape
        y (np array-like) MxP in shape
        destination (np array-like) optional location where to store the results as they are calculated (e.g. a np
            memmap of a file)

        returns (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provided, shape is NxP
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy
    """
    r = _fast_cov(np.mean, _fast_dot_divide, x, y, destination)

    return r


def _fast_cov(mean_method, dot_divide_method, x, y, destination):
    validate_inputs(x, y, destination)

    new_x = x if len(x.shape) == 2 else x[:, np.newaxis]

    if y is None:
        y = new_x
    new_y = y if len(y.shape) == 2 else y[:, np.newaxis]

    if destination is None:
        destination = np.zeros((new_x.shape[1], new_y.shape[1]))

    mean_x = mean_method(new_x, axis=0)
    mean_y = mean_method(new_y, axis=0)

    mean_centered_x = (new_x - mean_x).astype(destination.dtype)
    mean_centered_y = (new_y - mean_y).astype(destination.dtype)
    
    dot_divide_method(mean_centered_x, mean_centered_y, destination)

    return destination


def validate_inputs(x, y, destination):
    error_msg = ""

    if not hasattr(x, "shape"):
        error_msg += "x needs to be np array-like but it does not have \"shape\" attribute - type(x):  {}\n".format(type(x))
    
    if destination is not None and not hasattr(destination, "shape"):
        error_msg += "destination needs to be np array-like but it does not have \"shape\" attribute - type(destination):  {}\n".format(type(destination))

    if y is None:
        if destination is not None:
            expected_dim = x.shape[1] if len(x.shape) == 2 else 1
            expected_shape = (expected_dim, expected_dim)
            if destination.shape != expected_shape:
                error_msg += "x and destination provided, therefore destination must have shape matching number of columns of x but it does not - x.shape:  {}  expected_shape:  {}  destination.shape:  {}\n".format(
                    x.shape, expected_shape, destination.shape)
    else:
        if not hasattr(y, "shape"):
            error_msg += "y needs to be np array-like but it does not have \"shape\" attribute - type(y):  {}\n".format(type(y))
        elif x.shape[0] != y.shape[0]:
            error_msg += "the number of rows in the x and y matrices must be the same - x.shape:  {}  y.shape:  {}\n".format(x.shape, y.shape)
        elif destination is not None:
            expected_rows = x.shape[1] if len(x.shape) == 2 else 1
            expected_cols = y.shape[1] if len(y.shape) == 2 else 1
            expected_shape = (expected_rows, expected_cols)
            if destination.shape != expected_shape:
                error_msg += "x, y, and destination provided, therefore destination must have number of rows matching number of columns of x and destination needs to have number of columns matching number of columns of y - x.shape:  {}  y.shape:  {}  expected_shape:  {}  destination.shape:  {}\n".format(
                    x.shape, y.shape, expected_shape, destination.shape)

    if error_msg != "":
        raise CmapPyMathFastCovInvalidInputXY(error_msg)


def nan_fast_cov(x, y=None, destination=None):
    """calculate the covariance matrix (ignoring nan values) for the columns of x (MxN), or optionally, the covariance matrix between the
    columns of x and and the columns of y (MxP).  (In the language of statistics, the columns are variables, the rows
    are observations).

    Args:
        x (np array-like) MxN in shape
        y (np array-like) MxP in shape
        destination (np masked array-like) optional location where to store the results as they are calculated (e.g. a np
            memmap of a file)

        returns (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provided, shape is NxP
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy
    """
    x_masked = np.ma.array(x, mask=np.isnan(x))

    if y is None:
        y_masked = x_masked
    else:
        y_masked = np.ma.array(y, mask=np.isnan(y))

    dest_was_None = False
    if destination is None:
        num_rows = x_masked.shape[1] if len(x_masked.shape) == 2 else 1
        num_cols = y_masked.shape[1] if len(y_masked.shape) == 2 else 1
        destination = np.ma.zeros((num_rows, num_cols))
        dest_was_None = True

    r = _fast_cov(np.nanmean, _nan_dot_divide, x_masked, y_masked, destination)

    r[np.isinf(r)] = np.nan

    r = np.ma.filled(r, fill_value=np.nan) if dest_was_None else r

    return r


class CmapPyMathFastCovInvalidInputXY(Exception):
    pass


def fast_corr(x, y=None, destination=None):
    """calculate the pearson correlation matrix for the columns of x (with dimensions MxN), or optionally, the pearson correlaton matrix
    between x and y (with dimensions OxP).  If destination is provided, put the results there.  
    In the language of statistics the columns are the variables and the rows are the observations.

    Args:
        x (np array-like) MxN in shape
        y (optional, np array-like) OxP in shape.  M (# rows in x) must equal O (# rows in y)
        destination (np array-like) optional location where to store the results as they are calculated (e.g. a np
            memmap of a file)

        returns (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provied, shape is NxP
    
    References
    ----------
    [1] Math module at https://github.com/cmap/cmapPy
    """
    if y is None:
        y = x
    
    r = fast_cov(x, y, destination=destination)

    std_x = np.std(x, axis=0, ddof=1)
    if np.isscalar(std_x):
        std_x = np.array((std_x,))

    std_y = np.std(y, axis=0, ddof=1)
    if np.isscalar(std_y):
        std_y = np.array((std_y,))

    np.divide(r, std_x[:, np.newaxis], out=r)
    np.divide(r, std_y[np.newaxis, :], out=r)

    return r