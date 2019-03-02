
import os

import numpy as np
import matplotlib.pyplot as plt

from . import plot_helper

FIGSIZE=(3, 2.5)
FONTSIZE=12

def response(dpi, targetName, ylabel=None, savename=None):
    """
    Plot the stimulus and the corresponding response of the specified target.  Overlay the data with the model fit

    Input
    -----
    dpi : depictive class instance
    targetName : str
    ylabel : None or string
        if None, use the target name as y axis label, otherwise use given value
    savename : str

    """
    coluer = plot_helper.get_coluer(10, cmap='tab10')

    s = np.logspace(np.log10(dpi.get_stimulus().min()),
                    np.log10(dpi.get_stimulus().max()),
                    100)

    fig = plt.figure(figsize=FIGSIZE)

    plt.plot(dpi.get_stimulus(),
             dpi.get_target_data(targetName), 'o',
             ms=8, mfc='none',
             color=coluer[0, :], label='Data')

    plt.plot(s, dpi.predict_target(targetName, s=s),
            '-', color=coluer[1, :], label='Fit')
    plt.xscale('log')

    plt.xlabel('{} ({})'.format(dpi.get_stimulus_name(),
            dpi.get_stimulus_units()),
            fontsize=FONTSIZE)
    if ylabel is None:
        plt.ylabel(targetName, fontsize=FONTSIZE)
    else:
        plt.ylabel(ylabel, fontsize=FONTSIZE)

    plt.tight_layout()
    plot_helper.save(fig, savename)
    plt.show(block=False)
