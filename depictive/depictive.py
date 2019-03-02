import os
import re
import json

import numpy as np

from .meta_data import metaData
from .ssl import loadSSL
from .ssl import ssl


# =============================================================
# GENERIC METHODS
# =============================================================

class depictiveMethods:
    """
    Provides the methods for depictive analysis to either load and inference classes

    Methods
    -------
    - data access
        get_target_data
        get_target_names
        get_stimulus
        get_chan_names
        get_variance

    - stats subsection
        get_var_explained
        get_scaling
        predict_targets
        get_rsq

    - data persistance / printing
        to_file
        print_results
    """

    # ==================================================
    # ==================================================

    def get_stimulus(self):
        return self.model.stimulus

    # ==================================================
    # ==================================================

    def get_target_data(self, name):
        return self.model.get_targets(self.meta.get_target_idx(name))

    # ==================================================
    # ==================================================

    def get_target_names(self):
        return self.meta.get_target_names()

    # ==================================================
    # ==================================================

    def get_variance(self, name):
        idx = self.meta.get_chan_idx(name)
        return self.model.get_chan_variance(idx)
    # ==================================================
    # ==================================================

    def get_chan_names(self):
        return self.meta.get_chan_names()
    # ==================================================
    # ==================================================

    def get_stimulus_name(self):
        return self.meta.stim.name

    # ==================================================
    # ==================================================

    def get_stimulus_units(self):
        return self.meta.stim.units
    # ==================================================
    # ==================================================

    def get_var_explained(self, name):
        """
        Compute the variance explained by an observable by name
        Input:
            name : str
        Return
            float : variance explained
        """
        return self.model.get_var_explained(self.meta.get_chan_idx(name))

    # ==================================================
    # ==================================================

    def get_scaling(self, name):
        """
        Return scaling by observable name
        Input:
            name : str
        Return
            float : variance explained
        """
        return self.model.get_scaling(self.meta.get_chan_idx(name))

    # ==================================================
    # ==================================================

    def predict_target(self, targetName, s=None):
        """
        Input:
            name : str
                observable name
            s : ndarray
                default None : use experimental doses, otherwise
                (N dose, ) ndarray of stimulation strengths
        Return
            targets
        """
        if s is None:
            s = self.get_stimulus()
        return self.model.predict_target(s,
                self.meta.get_target_idx(targetName))

    # ==================================================
    # ==================================================

    def get_rsq(self):
        return self.model.get_rsq()

    # ==================================================
    # ==================================================

    def to_json(self, expName, savename):
        data = {expName : {
                    'components':{},
                    'doseResponse':{
                        'pars':{
                            'amplitude':self.model.get_hill_pars('amplitude'),
                            'sensitivity':self.model.get_hill_pars('sensitivity'),
                            'hillCoef':self.model.get_hill_pars('hillCoef'),
                            'background': self.model.get_hill_pars('background')
                        },
                        'stimulus': {
                            'name': self.get_stimulus_name(),
                            'units' : self.get_stimulus_units(),
                            'values' : self.get_stimulus().tolist()
                        },
                        'response':{
                            'name': self.meta.get_response_name(),
                            'values' : self.get_target_data(self.meta.get_response_name()).tolist()
                        }
                    },
                    'stats':{'rsq':self.get_rsq()},
                    'thresh':{'probThresh':self.model.thresh.threshCriterion,
                              'doseThresh':self.model.thresh.s_value}
                    }
                }

        for name in self.get_chan_names():
            data[expName]['components'][name] = {
                'target': self.get_target_data(name).tolist(),
                'scaling' : self.get_scaling(name),
                'varExplained': self.get_var_explained(name),
                'variance': self.get_variance(name)
            }

        with open(savename, 'w') as fid:
            fid.write(json.dumps(data, indent=2, sort_keys=True))
        return 0

    # ==================================================
    # ==================================================

    # def print_results(self):
    #     self.model.print_results()


# =============================================================
# LOAD
# =============================================================

class load(depictiveMethods):
    """
    Organize and run depictive analysis

    Properties
    ----------
    model


    Methods
    """
    meta = None
    model = None

    def __init__(self, jsonFileName):
        data = readJSONFile(jsonFileName)
        # self.meta = metaData()
        return 0

    def readJSONFile(self, jsonFileName):
        with open(jsonFileName, 'r') as fid:
            return json.loads(fid.read())

# =============================================================
# INFERENCE
# =============================================================

class inference(depictiveMethods):
    """
    Infer DEPICTIVE parameters and provide methods for DEPICTIVE analysis


    Properties
    ----------
    meta
    model

    """
    meta = None
    model = None

    def __init__(self, single_cell_data,
                    chanNames, responseName,
                    stimName, stimUnit,
                    thresh=0.9):
        '''
        Inputs
        ------
        single_cell_data : python list
            - semi supervised:
                N length list in which the i^th element is a dictionary:
                    {
                        'stimulus' : float, dose of stimulus
                        'data' : (n cell, m parameters) ndarray
                        'response' : float, fraction of cell's alive
                    }
                    corresponding to the ith stimulus dose
            - supervised
                N length list in which the i^th element is a dictionary:
                    {
                        'data': (n cell, m parameters) ndarray,
                        'response': (n cell, ) ndarray of binary values
                                    where 0 and 1 represent negative and positive class respectively
                    }
                    corresponding to the ith stimulus strength

        chanNames : list
            the ith element is the name of the ith biological observable
        stimName : str
            stimulus name
        stimUnit : str
            stimulus unit
        responseName : str

        Optional Input
        --------------
        thresh : float
            prevalence theshold in which samples whose corresponding prevalence > than thresh are assigned to the reference distribution, otherwise they are apart of the experimental group
        '''
        # store channel names, response names, units, etc.  Useful for plotting and refering mapping from names to idx
        self.meta = metaData(responseName,
                             chanNames,
                             stimName,
                             stimUnit)

        # apply relevant model, supervised or semi supervised
        if type(single_cell_data[0]['response']) == np.ndarray:
            raise ValueError('Error: Supervised learning is not yet implemented.')
            # self.model = supervised_learning(s, single_cell_data)
        else:
            self.model = ssl(single_cell_data, thresh)
