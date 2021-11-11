import numpy as np
from scipy import signal, fftpack


__all__ = ['butter_highpass',
           'butter_highpass_filter',
           'hp_filter_vibrations',
           'flattop_window',
           'scalloping_loss_corrected_fft',
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

    
def hp_filter_vibrations(x, y, fc_factor=5):
    """Highpass filter vibration data

    Parameters
    ----------
    x : 1D numpy array
        Time [s]
    y : 1D numpy array
        Vibration amplitude [nm]
    fc_factor : float or int
        Multiplication factor for high pass filter
        critical frequency. 

    Returns
    -------
    y : 1D numpy array of length data.shape[0]
        Filtered vibration amplitude [nm]

    """
    # Number of samplepoints
    N = len(y)
    # Sampling time
    T = x[1]-x[0]
    # Sampling frequency
    fs = 1/T
    # Frame time
    T_f = N*T
    # High pass critical frequency
    f_c = fc_factor * 1 / T_f
    # Filter data and return
    return butter_highpass_filter(y, f_c, fs)


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
    
    
def scalloping_loss_corrected_fft(y, T):
    N = len(y)
    yf = fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = 4.0/N * np.abs(flattop_window(yf)[:N//2])
    return xf, yf