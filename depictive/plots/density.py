
import numpy as np
import matplotlib.pyplot as plt

from . import plot_helper
from ..stats import hist


def density(x, y=None, reference=None,
        nbins=25, sname=None, xlabel=None):

    if xlabel is None:
        xlabel = 'x'

    if (y is not None) & (reference is not None):
        print('Please only provide reference distribution if all cells have a single class label')
    else:
        fig = plt.figure(figsize=(4.5, 4))

        if (y is None) & (reference is None):
            xtmp, ytmp = hist(x, nbins)
            plt.plot(xtmp, ytmp, '-', linewidth=2.5)
        elif y is not None:
            xtmp, ytmp = hist(x, nbins)
            plt.fill_between(xtmp, 0, ytmp, color='k', alpha=0.2,
                label='P({})'.format(xlabel))

            for wy in set(y):
                xtmp, ytmp = hist(x[y == wy], nbins)
                plt.plot(xtmp, ytmp, '-', linewidth=2.5,
                    label='P({}|y={})'.format(xlabel, wy))

            plt.legend(loc=0)
        else:
            xtmp, ytmp = hist(reference, nbins)
            plt.fill_between(xtmp, 0, ytmp, color='k', alpha=0.2,
                label='P({})'.format(xlabel))
            xtmp, ytmp = hist(x, nbins)
            plt.plot(xtmp, ytmp, '-', linewidth=2.5,
                label='P({}|y)'.format(xlabel))
            plt.legend(loc=0)

        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel('probability', fontsize=15)
        plt.tight_layout()
        plot_helper.save(fig, sname)
        plt.show(block=False)
