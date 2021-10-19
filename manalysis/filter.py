import numpy as np
from scipy import signal


__all__ = ['butter_highpass',
           'butter_highpass_filter',
           'flattop_window',
          ]


def butter_highpass(cutoff, fs, order=5):
    """Design an Nth-order digital or analog Butterworth filter 
    and return the filter coefficients.

    References
    ----------
    [1] https://stackoverflow.com/questions/39032325/python-high-pass-filter
    [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    

def butter_highpass_filter(data, cutoff, fs, order=5):
    """Filter data using Butter high pass filter

    References
    ----------
    [1] https://stackoverflow.com/questions/39032325/python-high-pass-filter
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def flattop_window(spectra):
    """Reducing FFT Scalloping Loss Errors Without Multiplication

    References
    ----------
    [1] DOI:10.1109/MSP.2010.939845
    """
    k = len(spectra)
    windowed_spectra = np.zeros_like(spectra)
    g_coeffs = [1, -0.94247, 0.44247]
    for i in range(0, k):
        windowed_spectra[i] = g_coeffs[2]*spectra[i-2] + \
                              g_coeffs[1]*spectra[i-1] + \
                              spectra[i] + \
                              g_coeffs[1]*spectra[(i+1) % k] + \
                              g_coeffs[2]*spectra[(i+2) % k]
    return windowed_spectra