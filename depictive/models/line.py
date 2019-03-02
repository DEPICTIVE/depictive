import numpy as np

class line:
    def __init__(self, x, y):
        self._fit(x, y)
        self._compute_rsq(x, y)

    def _fit(self, x, y):
        nans = x + y
        c = np.cov(x[~np.isnan(nans)], y[~np.isnan(nans)])
        self.slope = c[0, 1] / c[0, 0]

    def model(self, x):
        return line_function(self.slope, x)

    def _compute_rsq(self, x, y):
        errors = np.mean((y - self.model(x))**2)
        self.rsq = 1 - errors / np.var(y)

# =========================================
# =========================================

def line_function(k, x):
    return k * x
