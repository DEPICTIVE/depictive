import re
import numpy as np

from .data import threshold
from .data import reference
from .data import target

from .model import sslmodel
from ..stats import rsq

# ============================================================
# METHODS
# ============================================================

class sslMethods:
    """
    Methods
    -------
    - data access
        get_targets

    - stats
        get_rsq
        get_var_explained
        get_scaling
        get_hill_pars
        get_chan_variance

    - predictions
        predict_targets
        posterior
    """

    # =================================================
    # Get target one at a time
    # =================================================

    def get_chan_variance(self, idx):
        """
        Get channel variance from idx

        Input
        -----
        idx : int
            index of target to retrieve
        """
        return self.ref.C[idx, idx]

    # =================================================
    # Get target one at a time
    # =================================================

    def get_targets(self, idx):
        """
        Get target by idx
        Input
        -----
        idx : int
            index of target to retrieve
        """
        return self.targets.targets[:, idx]

    # =================================================
    # =================================================

    def get_rsq(self):
        return self.model.rsq

    # =================================================
    # =================================================

    def get_var_explained(self, idx):
        """
        Get the variance explained

        Input
        -----
        idx : int
            channel indexes
                [observable 1, observable 2, ..., observable M],
        """
        return self.model.get_var_explained(idx,
                self.ref.C[idx, idx])

    # =================================================
    # =================================================

    def get_scaling(self, idx):
        """
        Get the scaling

        Input
        -----
        idx : int
            idx is a number representing
                [observable 1, observable 2, ..., observable M],
        """
        return self.model.get_scaling(idx)

    # =================================================
    # =================================================

    def get_hill_pars(self, name=None):
        """
        Retrieve inferred Hill model parameters
        """
        if name is None:
            return self.model.get_hill_pars()
        else:
            hillPars = ['amplitude', 'sensitivity', 'hillcoef', 'background']
            name = name.lower()
            j = 0
            while (name != hillPars[j]) & (j < 4):
                j += 1
            if j < 4:
                return self.model.get_hill_pars()[j]
            else:
                raise ValueError('Error : {} is not a valid Hill par name'.foramt(name))

    # =================================================
    # =================================================

    def predict_target(self, s, idx):
        '''
        Model predicted class prevalence and conditional mean
        Inputs
        ------
        s: ndarray
            (N doses,) to estimate the specified target
        idx : int
            index of target to estimate
                [prevalence, chan 1, chan 2, ..., chan M]
        Returns
        -------
        - (N doses, ) the value of the desired target for each input dose
        '''
        if idx == 0:
            return self.model.infer_prevalence(s, self.ref.samples, self.ref.C)
        else:
            return self.model.infer_mean(s,
                self.ref.samples[:, idx-1].reshape(self.ref.shape[0], 1),
                self.ref.C[idx-1, idx-1],
                idx-1)
            # return self.model.infer_mean(s,
            #         self.ref.samples[:, idx-1].reshape(self.ref.shape[0], 1),
            #         self.ref.C[idx-1, idx-1], idx-1)

    # =================================================
    # =================================================

    def posterior(self, s, obs, idx):
        """
        Estimate the posterior distribution, P(response | s, obs[idx]) for different values of s or obs

        Input
        -----
        s : float or ndarray
            -float : compute the posterior given single dose and given obs values
            -ndarray: compute the posterior given obs value and various values of stimulus
        obs : float or ndarray
            -float: compute the posterior given obs value and various values of stimulus
            -ndarray: compute the posterior given single dose and given obs values
        idx : int
            representing the observable to condition on

        Return
        ------
        p : ndarray
            posterior evaluated accordingly
        """
        return self.model.posterior(s, obs, idx-1, self.ref.C[idx-1, idx-1])

# ============================================================
# LOAD parameters from file
# ============================================================

class loadSSL(sslMethods):
    """
    Set SSL properties from data load

    Properties
    ----------
    thresh : data.threshold class instance
    ref : data.reference class instance
    targets : data.target class instance
    fit : fit.ssl class
    """

    def __init__(self):
        return 0

# ============================================================
# INFER parameters from data
# ============================================================

class ssl(sslMethods):
    '''
    Semi-supervised inference of Hill model and scaling parameters from single cell measurements

    Properties
    ----------
    stimulus: ndarray
    thresh : data.threshold class instance
    ref : data.reference class instance
    tdata : data.target class instance
    fit : fit.ssl class

    '''
    stimulus = None
    thresh = threshold()
    ref = reference()
    targets = target()
    model = sslmodel()

    # ==========================================
    # ==========================================

    def __init__(self, data, thresh):
        """
        Inputs
        ------
        data : list
            See DEPICTIVE.inference for description
        thresh: float
            The probability of cells being alive given TRAIL criterion for reference samples
        """
        # decompose entered data
        self.stimulus, scd, response = self._decompose_inference_data_input(data)

        # organize data set
        self.thresh.set(self.stimulus, response, thresh)
        self.ref.set(scd, self.thresh)
        self.targets.set(scd, response, self.thresh, self.ref)

        # fit data using ssl model
        self.model.fit(self.stimulus,
                self.ref,
                self.targets.targets)

    # =================================================
    # =================================================

    def _decompose_inference_data_input(self, data):
        """
        Input
        -----
        data : python list
            i^th entry contains dictionary of experimental data, see depictive for description

        Returns
        -------
        s : ndarray
            (N doses, ) ordered stimulus dose, ordered weak to strong
        scd : list
            each entry i contains an (N cell, M observable) ndarray corresponding to the i^th stimulus dose
        r : ndarray
            (N doses, ) measured population response, ordered by s
        """
        s = np.zeros(len(data))
        r = np.zeros(len(data))
        scd = list(range(len(data)))
        j = 0
        for w in data:
            s[j] = w['stimulus']
            r[j] = w['response']
            if w['data'].size == w['data'].shape[0]:
                scd[j] = w['data'].reshape(w['data'].size, 1)
            else:
                scd[j] = w['data']
            j += 1
        idx = np.argsort(s)
        return [s[idx], [scd[i] for i in idx], r[idx]]

    # =================================================
    # =================================================

    # def get_pars(self, parName=None):
    #     parDict = {'amplitude': self.pars[0],
    #                 'ic50': self.pars[1],
    #                 'hillCoef': self.pars[2],
    #                 'background':self.pars[3]}
    #     for j in range(4, self.pars.size):
    #         parDict['scaling_' + str(j-3)] = self.pars[j]
    #     if parName is None:
    #         return parDict
    #     else:
    #         return parDict[parName]

    # =================================================
    # =================================================

    # def get_beta(self):
    #     return ssl.estimate_beta(self.pars, self.C)
    #
    # # =================================================
    # # =================================================
    #
    # def get_chan_beta(self):
    #     pars = self.get_pars()
    #     beta = {}
    #     for j in range(4, self.pars.size):
    #         scaleName = 'scaling_{}'.format(j-3)
    #         tmpPars = [pars['amplitude'], pars['ic50'], pars['hillCoef'],
    #                     pars['background'], pars[scaleName]]
    #         beta[scaleName] = ssl.estimate_beta(tmpPars, self.C[j-4, j-4])
    #     return beta
    #
    # # =================================================
    # # =================================================
    #
    # def print_results(self):
    #     pars = self.get_pars()
    #     beta = self.get_chan_beta()
    #     vars = beta.copy()
    #     for w in beta:
    #         vars[w] = 1 - (pars['hillCoef'] / beta[w])**2
    #     print('Inferred parameters')
    #     for par_key in pars:
    #         if re.search('scaling', par_key) is None:
    #             print('{} \t:\t {:0.3f}'.format(par_key, pars[par_key]))
    #         else:
    #             print('{} \t:\t {:0.3f} \t VarExplain {:0.3f}'.format(par_key,
    #                     pars[par_key], vars[par_key]))
    #     print('Rsq : {:0.3f}'.format(self.rsq))

# =============================================================
# Opimization functions, required for method, but not included as class methods
# =============================================================
# =============================================================
