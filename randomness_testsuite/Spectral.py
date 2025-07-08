from math import fabs, floor, log, sqrt
from numpy import where, array
from scipy import fftpack as sff
from scipy.special import erfc

class SpectralTest:
    @staticmethod
    def spectral_test(data, verbose=False):
        # Accepts either binary string or list/array of 0/1 or ±1
        if isinstance(data, str):
            if len(data) < 64:
                return 0.0, False
            data = [-1 if c == '0' else 1 for c in data if c in '01']
        else:
            data = array(data)
            if len(data) < 64:
                return 0.0, False
            if not set(data).issubset({-1, 0, 1}):
                raise ValueError("Numeric input must contain only -1, 0, or 1")
            data = 2 * data - 1  # convert 0/1 to -1/+1 if needed

        n = len(data)
        spectral = sff.fft(data)
        modulus = abs(spectral[: n // 2])
        tau = sqrt(log(1 / 0.05) * n)
        n0 = 0.95 * (n / 2)
        n1 = len(where(modulus < tau)[0])
        d = (n1 - n0) / sqrt(n * 0.95 * 0.05 / 4)
        p_value = erfc(fabs(d) / sqrt(2))

        if verbose:
            print("Spectral Test Debug Info")
            print(f"Length: {n}, T: {tau}, n0: {n0}, n1: {n1}, d: {d}, p: {p_value}")

        return float(p_value), bool(p_value >= 0.01)
