
import numpy as np

from ..models import hill

# ====================================================
# REFERENCE CLASS
# ====================================================

class reference:
    """
    Organize reference samples

    Properties
    ----------
    shape: tuple
        (Nref cells, M observables)
    samples : ndarray
        (Nref cells, M observables) single cell measurements
    C : ndarray
        (M observables, M observables) Covariance matrix
    mean : ndarray
        (M observables, ) un-normalized observable means,


    Methods
    -------
    set_from_JSON_load
    set
    _set_reference_samples
    _set_ref_stats
    norm_samples
    """
    shape = None
    samples = None
    C = None
    mean = None

    # ==========================================
    # ==========================================

    def set_from_JSON_load(self):
        return 0

    # ==========================================
    # ==========================================

    def set(self, scd, thresh):
        """
        Estimate values necessary for setting properties

        Input
        -----
        scd : list
        thresh : object
            instance of thesh class
        """
        self._set_reference_samples(scd, thresh)
        self._set_ref_stats()
        self.samples = self.norm_samples(self.samples)

    # ==========================================
    # ==========================================

    def _set_reference_samples(self, scd, thresh):
        """
        Establish reference single cell measuresmetns by, aggregate samples according to thresh idx

        Input
        -----
        scd : list
        thresh : thresho object
        """
        self.samples = scd[0]
        for j in range(1, thresh.idx):
            self.samples = np.vstack([self.samples, scd[j]])

    # ==========================================
    # ==========================================

    def _set_ref_stats(self):
        self.shape = self.samples.shape
        self.mean = np.mean(self.samples, 0)
        self.C = np.diag(np.var(self.samples, 0))

    # ==========================================
    # ==========================================

    def norm_samples(self, data):
        """
        Normalize samples by zero mean according to reference stats

        Input
        -----
        data : ndarray
            (N samples, M observables) of single cell measurements
        """
        return data - np.tile(self.mean.reshape(1, self.shape[1]), (data.shape[0], 1))

# ====================================================
# THRESHOLD CLASS
# ====================================================

class threshold:
    """
    Organize threshold information for defining the reference and target data

    Properties
    -----
    threshCriterion
    s_value
    idx


    Method
    ------
    set_from_JSON_load
    set
    _set_thresh_idx
    _compute_thresh
    """
    threshCriterion = None
    s_value = None
    idx = None

    # ==========================================
    # ==========================================

    def set_from_JSON_load(self):
        return 0

    # ==========================================
    # ==========================================

    def set(self, stimulus, response, threshCriterion):
        """
        Input
        -----
        stimulus : ndarray
            (N doses,) stimulus array
        response : ndarray
            (N doses, ) response array
        threshCrietiron : float
            the response used for separating reference and target samples
        """
        self.threshCriterion = threshCriterion
        h = hill.fit(stimulus, response)
        self.s_value = self._compute_thresh(h.pars)
        self.idx = self._set_thresh_idx(stimulus)

    # ==========================================
    # ==========================================

    def _set_thresh_idx(self, stimulus):
        '''
        Find idx threshold, stimulant doses below s_value are condidered apart of reference dist

        Input
        -----
        stimulus : ndarray
            (N doses, ) array

        Return
        ------
        j : int
            the index representing the smallest stimulus dose that is greater than s_value
        '''
        j = 0
        while stimulus[j] < self.s_value:
            j += 1
        return j

    # ==========================================
    # ==========================================

    def _compute_thresh(self, pars):
        '''
        Estimate s_value, the thrshold stimulus dose, using the user inputed stimulus response threshold and the Hill model.  Specifically, let y = f(x), where f(x) is the Hill model

        s_value = k*(1/threshCriterion - 1)**(1/n)

        Input
        -----
        pars : ndarray
            Parameters of the hill model,
                [Amplitude, half max constant, Hill coefficient, background]

        Returns
        -------
        float
            the dose of stimulus corresponding to the response threshCriterion.
        '''
        return pars[1]*(1/self.threshCriterion - 1)**(1/pars[2])

# ====================================================
# Single cell data set
# ====================================================

class target:
    """
    Organize targets for fitting DEPICTIVE strategy

    Properties
    ----------
    M : int
        Number of observables
    targets : ndarray
        (N doses, M observables + 1)
        First column is always the response
        Columns i+1, correspond to the conditional averages for the i^th observable
    """
    M = None
    targets = None

    # ==========================================
    # ==========================================

    def set_from_JSON_load(self):
        return 0

    # ======================================
    # ======================================

    def set(self, single_cell_data, response, thresh, ref):
        """
        Input
        -----
        single_cell_data : list
            the i^{th} list entry is (N cells, M observables) ndarray corresponding to the i^th stimulus dose
        response : ndarray
            (N doses, ) respose to each dose of stimulant
        ref : reference class instance
        thresh : threshold class instance

        """
        self.M = ref.shape[1]
        self.targets = None
        # self._set_samples(scd, thresh, ref)
        self._set_targets(single_cell_data, response, thresh, ref)

    # ======================================
    # ======================================

    def _set_targets(self, scd, response, thresh, ref):
        """
        Compute targets of experimental samples from single cell data

        Input
        -----
        scd : list
        response : ndarray
        thresh : threshold class instance
        ref : reference class instance
        """
        self.targets = np.zeros(shape=(response.size, self.M+1))
        # the first column of targets is always the response
        self.targets[:, 0] = response
        for j in range(thresh.idx, response.size):
            self.targets[j, 1:] = np.mean(ref.norm_samples(scd[j]), 0)
