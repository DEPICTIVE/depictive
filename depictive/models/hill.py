import numpy as np
from scipy.optimize import fmin

class fit:
    '''
    Infer the Hill model parameters from data:
    Hill pars:
    0) amplitude
    1) ic50
    2) Hill coefficient
    3) background
    '''
    def __init__(self, x, y, max_iter=5e4,
            max_fun=5e4, disp=False, ko=False):
        self.max_iter = max_iter
        self.max_fun = max_fun
        self.disp = disp
        self.ko = ko
        self.pars = self._fit(x, y)

    def _fit(self, x, y):
        if self.ko is False:
            self.ko = _get_hill_ko(x, y)
        return fmin(_sse, self.ko, args=(x, y),
            maxfun=self.max_fun, maxiter=self.max_iter, disp=self.disp)

    def model(self, x):
        return hill_function(self.pars, x)

    def standardized_model(self, x):
        return hill_function([1., self.pars[1], self.pars[2], 0.], x)


# =========================================
# =========================================

def _get_hill_ko(x, y):
    return [y.max() - y.min(),
            np.exp(np.mean(np.log(x))),
            0.5,
            y.min()]

# =========================================
# =========================================

def hill_function(k, x):
    return k[0] / (1 + (x/k[1])**k[2]) + k[3]

# =========================================
# =========================================

def _sse(k, x, y):
    penalty = 0
    # amplitude contraints
    if (k[0] + k[-1] > 1) | (k[0] + k[-1] < y.max()):
        penalty =1e4
    # background constraint
    if (k[-1] < 0) | (k[-1] > y.min()):
        penalty =1e4
    if (k[2] > 10) | (k[2] < 0):
        penalty =1e4
    return np.sum((y - hill_function(k, x))**2) + penalty
