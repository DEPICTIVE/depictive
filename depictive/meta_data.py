
import numpy as np

class stim:
    """
    Organize stimulus meta data

    Properties
    ----------
    name
    units

    """
    def __init__(self, stimName, stimUnits):
        self.name = stimName
        self.units = stimUnits
        if stimName is None:
            self.name = 's'
        if stimUnits is None:
            self.units = 'a.u.'

# ============================================================
# ============================================================

class metaData:
    """
    Organize meta data

    Properties
    ----------
    targetNames
    stim
    M

    METHODS
    -------
    get_chan_idx
    get_chan_names
    get_target_idx
    get_target_names
    """
    chans = None
    M = None
    stim = None

    def __init__(self, responseName, chanNames, stimName, stimUnit):
        """
        Organize names of targets and stimulus

        Input
        -----
        responseName : str
            the name of the response in which observables are attributed to
        chanNames: list
            the ith entry is the name of the ith observable
        stimName : str
            stimulus name
        stimUnit : str
            stimulus units
        """
        self.targetNames = [responseName]
        if (type(chanNames) != list) & (type(chanNames) != np.ndarray):
            self.targetNames.append(chanNames)
        else:
            self.targetNames += chanNames
        self.M = len(self.targetNames)
        self.stim = stim(stimName, stimUnit)

    # ========================================
    # ========================================

    def get_target_names(self):
        return self.targetNames

    # ========================================
    # ========================================

    def get_target_idx(self, name):
        """
        Get idx corresponding to channel name

        Input
        -----
        name : str
            name of channel whose index is requested

        Return
        ------
        j : int
            channel index
        """
        j = 0
        while (self.targetNames[j] != name) & (j < self.M):
            j += 1
        if j == self.M:
            raise ValueError('Error : {} is not a valide channel name'.format(name))
        return j
    # ========================================
    # ========================================

    def get_response_name(self):
        return self.targetNames[0]

    # ========================================
    # ========================================

    def get_chan_names(self):
        return self.targetNames[1:]

    # ========================================
    # ========================================

    def get_chan_idx(self, name):
        return self.get_target_idx(name) - 1

    # ========================================
    # ========================================
