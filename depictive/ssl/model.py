
import numpy as np
from scipy.optimize import fmin

from ..models import hill
from ..stats import rsq


# ==================================================================
# METHODS
# ==================================================================

class sslModelMethods:
    """
    Methods for semi-supervised learning

    get_var_explained
    get_scaling
    get_hill_pars
    get_pars

    infer_prevalence
    infer_mean

    posterior
    _cond_dose_response
    _posterior_given_stim
    """
    # =================================
    # =================================

    def get_var_explained(self, idx, v):
        """
        Compute the variance explained by observable idx

        Input
        -----
        idx : int
            index of observable whose variance explain is requested, [0, M-1]
        v : float
            the reference distribution variane of component idx
        """
        return 3 * self.get_scaling(idx)**2 * v * self.pars[2]**2 / np.pi**2


    # =================================
    # =================================

    def get_scaling(self, idx):
        """
        Input
        -----
        idx : int
            index of observable whose scaling is requested, [0, M-1]

        Return
        ------
        float
            scaling parameter
        """
        return self.pars[idx + 4]
    # =================================
    # =================================

    def get_hill_pars(self):
        """
        Return
        -----
        ndarray
            Hill parameters [Amplitude, half max constant, Hill coef, background]
        """
        return self.pars[:4]

    # =================================
    # =================================

    def get_pars(self, idx):
        """
        Construct parameter set for the idx^{th} observable

        Input
        -----
        idx : int
            index of requested observable, [0, M-1]

        Return
        ------
        ndarray
            Hill parameters, idx scaling
            [Amplitude, half max constant, Hill coefficient, background, idx^th scaling constant]
        """
        return np.hstack([self.pars[:4], self.pars[4+idx]])

    # =================================
    # =================================

    def infer_prevalence(self, stimulus, refSamples, C):
        """
        Predict the prevalence of the positive class, otherwise called response fraction

        Input
        -----
        stimulus
        refSamples
        C
        """
        predict_prevalence = np.zeros(stimulus.size)
        for j in range(stimulus.size):
            predict_prevalence[j] = infer_prevalence(self.pars,
                                        stimulus[j],
                                        refSamples, C)
        return predict_prevalence

    # =================================
    # =================================

    def infer_mean(self, stimulus, refSamples, C, idx):
        """
        Input
        -----
        stimulus : ndarray
            (N doses, ) of stimulus strengths
        refSamples : ndarray
            (N cells, 1) of single cell reference sample measurements
        C : float
            variance of the desired component
        idx : int
            index of for the requested observable [0, M-1]

        Return
        ------
        predict_mean : ndarray
            (N doses, ) of conditional averages of the idx^{th} observable
        """
        predict_mean = np.zeros(stimulus.size)
        parameters = self.get_pars(idx)
        for j in range(stimulus.size):
            p = model(parameters, stimulus[j], refSamples, C)
            predict_mean[j] = np.mean(refSamples.squeeze() * p) / np.mean(p)
        return predict_mean

    # =================================
    # =================================

    def posterior(self, s, obs, idx):
        """
        Compute the posterior distribution of live cells given stimulus dose and observable values.

        Input
        s : float or ndarray
            float, (ndarray) : value(s) of stimulus
        obs : float or ndarray
            float, (ndarray) : value(s) of observable corresponding to idx
        idx : int
            Channel index
        """
        if type(s) == np.ndarray:
            return self._cond_dose_response(s, obs, idx)
        else:
            return self._posterior_given_stim(s, obs, idx)

    # =================================
    # =================================

    def _cond_dose_response(self, s, obs, idx, C):
        """
        """
        # instantiate energy
        U = np.zeros(s.size)
        # subselec parameter sets
        params = self.get_pars(idx)
        # estimate beta
        beta = estimate_beta(pars, C)
        U -= params[-1] * obs
        for j in range(s.size):
            U[j] += np.log(s[j] / params[1])
        return (1 + np.exp(beta * U))**-1

    # =================================
    # =================================

    def _posterior_given_stim(self, s, obs, idx):
        """
        """
        U = np.zeros(obs.size)
        # subselect parameter sets
        params = self.get_pars(idx)
        # estimate beta
        beta = estimate_beta(pars, C)
        # start U
        U += np.log(s / params[1])
        for j in range(obs.size):
            U[j] -= params[-1] * obs
        return (1 + np.exp(beta * U))**-1

# ==================================================================
# LOAD
# ==================================================================

class loadModel(sslModelMethods):
    def __init__(self, jsonFileName):
        return 0

# ==================================================================
# INFERENCE
# ==================================================================

class sslmodel(sslModelMethods):
    """
    Properties
    ----------
    pars : ndarray
        (M observables + 4),
            [Amplitude, Half Max Constant, Hill Coef, Background,
                scaling 1, scaling 2, ..., scaling M]
    rsq : float
        Coefficient of determination
    max_iter : integer
        (default 5000)
    tol : float
        (default 1e-8)

    Methods
    -------
    set_from_JSON_load
    fit

    _compute_coef_determination


    """
    pars = None
    rsq = None
    max_iter=5000
    tol=1e-8

    # =================================
    # =================================

    def fit(self, stimulus, ref, targets):
        """
        Run fitSSL and store / organize parameters

        Input
        -----
        stimulus : ndarray
            (N doses)
        ref : reference class instance
        targets : ndarray
            (N doses, M observabels + 1) array of targets to fit
        """
        self.pars, fval = fitSSL(stimulus, ref.samples, targets, ref.shape[1], ref.C)
        self.rsq = self._compute_coef_determination(stimulus, ref.samples, targets, ref.C)

    # =================================
    # =================================

    def _compute_coef_determination(self, stimulus, refSamples, targets, C):
        """
        Compute the coefficient of determination as a measure of goodness of fit

        Inputs
        ------
        stimulus : ndarray
            (N doses, ) stimulant doses
        refSamples : ndarray
            (N cells, M observables)
        targets : ndarray
            (N doses, M observables + 1)
        C : ndarray
            (M observable, M observable) covariance matrix

        Return
        ------
        coefficient of determination : float
        """
        # instantiate predictions array, first column is prevalence, the following are conditional averages
        infer_targets = np.zeros(targets.shape)
        # iterate over stimulus strength
        for j in range(stimulus.size):
            infer_targets[j, 0] = infer_prevalence(self.pars, stimulus[j], refSamples, C)
            infer_targets[j, 1:] = infer_mean(self.pars, stimulus[j], refSamples, C)
        return rsq(targets.ravel(), infer_targets.ravel())


# ===============================================================
# semi-supervised fitting
# ===============================================================



def fitSSL(stimulus, ref_samples, targets, M, C, max_iter=5000, tol=1e-8):
    '''
    Semi-supervised logistic regression for analysis of dose response data

    Inputs
    -----
    stimulus : ndarray
        (N doses,) numpy array of stimulant doses
    ref_samples : ndarray
        (Nref cells, M observables) numpy array
    targets : ndarray
        (s stimuli, M+1 observables) numpy array
            - column 1 is the fraction of cells alive
            - column 2:M+1 is the emprical average of each observable given samples are from the positive class.(N doses, M+1 statistical quantities)
    M : int
        Number of observables
    C : ndarray
        (M observable, M observable) covariance matrix
    max_iter : int
        (optional)
    tol : float
        (optional)

    Returns
    -------
    [inferred parametes, fval]

    '''
    # fit zeroth order moment to Hill model
    h = hill.fit(stimulus, targets[:, 0])

    # fit scaling parameter
    out = fmin(_scaling_obj, np.zeros(M),
                args=(stimulus, ref_samples, C,
                targets[:, 1:], h.pars), disp=False,
                full_output=1)

    # store pars
    pars = np.hstack([h.pars, out[0]])
    error = np.array([4, out[1]])
    j = 2
    while (j < max_iter) & (np.abs(error[1]-error[0]) > tol):
        error[1] = error[0]
        # fit zeroth order moment to Hill model
        out = fmin(_hill_obj, pars[:4],
            args=(stimulus, ref_samples, C,
                    targets[:, 0], pars[4:]),
            full_output=True, disp=False)
        pars[:4] = out[0]
        error[0] = out[1]

        # infer scaling parameter using first moment measurements
        out = fmin(_scaling_obj, pars[4:],
                args=(stimulus, ref_samples, C,
                targets[:, 1:], pars[:4]), disp=False,
                full_output=1)

        # store the inferred parameters and error
        pars[4:] = out[0]
        error[0] += out[1]
        j+=1
    if j == max_iter:
        print('Did Not Converge')
    return [pars, error]


# ==========================================
# objective for fitting the average
# ==========================================

def _scaling_obj(k, stimulus, refs, C, targets, hill_pars):
    '''
    Objective function for finding scaling parameters

    Inputs
    ------
    k : ndarray
        (M observable, ) numpy arrray [scaling 1, scaling 2, ..., scaling M]
    stimulus : ndarray
        (N doses, ) numpy array of stimulant doses
    ref : ndarray
        (Nref cells, M observables) numpy arrayn
    C : ndarray
        (M obs, M obs) Empirical covariance matrix of observables
    targets : ndarray
        (N doses, M observables)
    hill_pars : ndarray
        parameters of hill model

    Returns
    -------
    scalar
    '''
    inferred_mean = np.zeros(shape=(stimulus.size, C.shape[0]))

    parameters = np.hstack([hill_pars, k])
    # infer means
    for j in range(stimulus.size):
        inferred_mean[j, :] = infer_mean(parameters,
                                    stimulus[j],
                                    refs, C)
    # sum errors over stimuli
    error = np.abs(inferred_mean - targets)
    return np.sum(error.ravel())

# ==========================================
# objective for fitting the positive class prevalence
# ==========================================

def _hill_obj(k, stimulus, ref_samples,
              C, targets, scaling_pars):
    '''
    Objective function for finding hill parameters
    Inputs
    ------
    k : ndarray
        [amp, ic50, hill coef, background, scaling i]
    stimulus : ndarray
        (N doses, )
    ref_samples : ndarray
        (n cells, M observables) reference samples
    C : ndarray
        (M observables, M observables)
    targets : ndarray
        (N doses,) the empirical fraction of cells alive with dose
    scaling_pars : ndarray
        (M observables, ) the scaling parameters

    Returns
    -------
    scalar
    '''
    penalty = 0
    # max value needs to be less than 1 but larger than the maximum value of the data
    if (k[0] + k[3] > 1) | (k[0] + k[3] < targets.max()):
        penalty += 1000
    # the hill coefficient needs to be between 0 and 10
    if (k[2] < 0) | (k[2] > 10):
        penalty += 1000
    # the background must be between the smallest data value and zero
    if (k[3] > targets.min()) | (k[3] < 0):
        penalty += 1000

    # instantiate mean prediction array
    inferred_prevalence = np.zeros(stimulus.size)
    parameters = np.hstack([k, scaling_pars])
    # loop over stimulus strengths
    for j in range(stimulus.size):
        inferred_prevalence[j] = infer_prevalence(parameters,
                                    stimulus[j],
                                    ref_samples,
                                    C)

    err = np.sum(np.abs(targets - inferred_prevalence))
    return err + penalty

# ===============================================================
# estimage the inverse temperature
# ===============================================================

def estimate_beta(k, C):
    '''
    Estimate the inverse temperature from scaling parameters and variances
    Inputs
    ------
    k: ndarray
        [amp, ic, ensemble hill coef, background,
            scaling i]
    C: ndarray
        (M observable, M observable) covariance matrix

    Returns
    -------
    scalar
    '''
    sig = np.pi**2 / (3*k[2]**2)
    if type(C) == np.ndarray:
        K = k[4:].reshape(C.shape[0], 1)
        condSigmaK = sig - np.dot(K.T, np.dot(C, K)).squeeze()
    else:
        condSigmaK = sig - k[4]**2 * C
    return np.pi / (np.sqrt(3*condSigmaK))

# ===============================================================
# complete model
# ===============================================================

def model(k, s, x, C):
    '''
    Probability of class 1 given parameters, stimuli, sample observable and variance

    Input
    -----
    k : ndarray
        [amp, ic, ensemble hill coef, background,
            scaling 1, scaling 2, ...]
    s: float
        scalar of stimuli strength
    x: ndarray
        (N samples, M observables) cell measurements from reference distribution
    C: float or ndarray
        observable variance, or
        (M observables, M observables) covariance matrix


    Return
    --------
    ndarray
        (N samples,) probabilities
    '''
    beta = estimate_beta(k, C)
    U = np.log(s/k[1])
    for j in range(x.shape[1]):
        U -= k[4 + j] * x[:, j]
    return (1 + np.exp(beta * U))**-1

# ==========================================
# compute prevalence
# ==========================================

def infer_prevalence(k, stimulus, refs, C):
    '''
    Compute the zeroth moment of the model from references samples
    Inputs
    ------
    k : ndarray
        [amp, ic, ensemble hill coef, background, scaling 1, scaling 2, ...]
    stimulus : float
        scalar of stimulus magnitude
    ref: ndarray
        (N samples, M observables) Numpy array of reference samples
    C : ndarray
        (M observables, M observables) Covaraince matrix

    Returns
    -------
    scalar
    '''
    return k[0]*np.mean(model(k, stimulus, refs, C)) + k[3]

# ==========================================
# compute conditional means
# ==========================================

def infer_mean(k, stimulus, ref, C):
    """
    Compute the conditional mean for each observable

    Inputs
    ------
    k : ndarray
        (4 + M, )
        - [amp, ic, ensemble hill coef, background,
            scaling 1, scaling 2, ...]
    stimulus : float
        scalar of stimulus magnitude
    ref: ndarray
        (N samples, M observable) Numpy array of reference samples
    C: ndarray
        (M observable, M observable) Empirical covariance matrix of ref. observations

    Returns
    -------
    (M observable,) Numpy array
    """
    target = np.zeros(C.shape[0])
    for j in range(C.shape[0]):
        parameters = np.hstack([k[:4], k[4 + j]])
        p = model(parameters, stimulus,
                ref[:, j].reshape(ref.shape[0], 1),
                C[j, j])
        target[j] = np.mean(ref[:, j] * p) / np.mean(p)
    return target
