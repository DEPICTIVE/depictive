
import numpy as np
from scipy.stats import gaussian_kde as kde
from scipy.integrate import trapz


def density(data, x=False, n=100):
    kernel = kde(data)

    if x:
        return kernel(x)
    else:
        x = np.linspace(x.min(), x.max(), n)
        return [x, kernel(x)]


def hist(data, nbins):
    y, x = np.histogram(data, bins=nbins)
    x = x[1:] - 0.5*(x[1] - x[0])
    return [x, y / trapz(y, x)]
