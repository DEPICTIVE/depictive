

import numpy as np
import matplotlib.pyplot as plt

from . import plot_helper

FONTSIZE=12


def coefficientDetermination(dp, label={}, savename=None):
    """
    Plot the empirical and inferred targets and Rsq

    Input
    -----
    dp : depictive class instance
    label : dict
    savename : None or string
        string to save figure to
    """
    coluer = plot_helper.get_coluer(20, cmap='tab20')

    lims = [10, 0]

    fig = plt.figure(figsize=(3.75, 2.5))
    j = 0
    for name in dp.get_target_names():
        # get empirical values
        tmp = dp.get_target_data(name)
        # read min maxes for setting line y = x
        if tmp.min() < lims[0]:
            lims[0] = tmp.min()
        if tmp.max() > lims[1]:
            lims[1] = tmp.max()

        # plot targets vs inference
        if name in label:
            plt.plot(tmp, dp.predict_target(name), ':o',
                 mfc='none', mew=2, ms=10, color=coluer[j, :],
                 label=label[name])
        else:
            plt.plot(tmp, dp.predict_target(name), ':o',
                    mfc='none', mew=2, ms=10, color=coluer[j, :],
                    label=name)
        j += 1
    plt.plot(lims, lims, ':', color='k', alpha=0.25, label='y=x')
    plt.text(lims[0], lims[1],
            '{} : {:0.3f}'.format(r'$R^{2}$', dp.get_rsq()),
            verticalalignment='top',
            horizontalalignment='left')
    ax = plt.gca()
    ax.set_position([0.15, 0.175, 0.45, 0.775])
    plt.xlabel('True', fontsize=FONTSIZE)
    plt.ylabel('Inferred', fontsize=FONTSIZE)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plot_helper.save(fig, savename)
