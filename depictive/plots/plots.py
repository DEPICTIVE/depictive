
import numpy as np
import matplotlib.pyplot as plt

from . import plot_helper

def l1_sims(depict, labels=None, xlabel=None, sname=None, cmap='tab20'):
    if type(depict) != list:
        depict = [depict]

    if labels is None:
        labels = [None for w in range(len(depict))]

    coluer = plot_helper.get_coluer(len(depict), cmap=cmap)

    fig = plt.figure(figsize=(4.5, 4))
    count = 0
    for wdepict in depict:
        plt.plot(wdepict.get('s'), wdepict.get('l1_sim'),
            ':o', mfc='none', color=coluer[count, :], ms=12.5,
            linewidth=1.5, mew=1.5, label=labels[count])
        count += 1
    if labels[0] is not None:
        plt.legend(loc=0, fontsize=15)
    plt.xscale('log')
    if xlabel is None:
        xlabel='dose'
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('L1 Similarity', fontsize=15)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plot_helper.save(fig, sname)
    plt.show(block=False)
