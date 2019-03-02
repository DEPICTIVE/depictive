
import numpy as np
import matplotlib.pyplot as plt

from . import plot_helper

def posterior(dclass, xlims,
            cmap='viridis', sname=None, xlabel=None):
    if xlabel is None:
        xlabel = 'x'

    xplot = np.linspace(xlims[0], xlims[1], 100)
    idx = np.argsort(dclass.get('s'))
    coluer = plot_helper.get_coluer(idx.size, cmap=cmap)

    fig = plt.figure(figsize=(4.5, 4))
    count = 0
    for widx in idx:
        plt.plot(xplot, dclass.l[widx].model(xplot), '-',
                    linewidth=2.5,
                    color=coluer[count, :])
        count += 1

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('P(alive | {}, s)'.format(xlabel), fontsize=15)
    plt.tight_layout()

    plot_helper.save(fig, sname)
    plt.show(block=False)
