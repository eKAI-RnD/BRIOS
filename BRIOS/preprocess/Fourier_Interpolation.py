import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, ifft


class FourierInterpolation:
    def __init__(self, series, original_dates, new_dates):
        self.series = series
        self.original_dates = original_dates
        self.new_dates = new_dates

    def interpolation(self):
        ndvi_fft = fft(self.series)
        n_original = len(self.original_dates)
        n_interp = len(self.new_dates)

        fft_padded = np.pad(ndvi_fft, (0, n_interp - n_original), mode='constant')
        interp_values = ifft(fft_padded).real

        return interp_values

