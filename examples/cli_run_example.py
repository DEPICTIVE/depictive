
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

from depictive import inference
from depictive import plots
from depictive.simulate import sample

# from depictive.plots import dose_response
# from depictive.plots import moments

PARS = [0.25, 0.5, 1., 1.5, 2.]
NCELLS = 10000
NDOSES = 13
IDX = [1, 2, 3]
# ========================================
# MAKE SYNTHETIC DATA
# ========================================

def get_data(pars, ndoses, ncells, idx):
    """
    Make synthetic data set
    pars : list
        simulation parameters
    ndoses : int
        number of doses
    ncells : int
        number of cells
    idx : list or ndarray
        the indexes of the "observable" subset of constituents
    """
    print('===========================')
    print('Simulating')
    s, X, labels = sample(pars,
                          Ncells=ncells,
                          n_doses=ndoses)
    # fraction of cells alive
    falive = np.mean(labels, 0)
    data = list(range(s.size))
    for j in range(s.size):
        data[j] = {'stimulus':s[j],
                   'response':falive[j]}
        data[j]['data'] = X[labels[:, j] == 1, :, j][:, idx]
    return data

# ========================================
# MAIN
# ========================================

def main(pars, ndoses, ncells, idx):
    # =========================
    # simulate
    # =========================

    data = get_data(pars, ndoses, ncells, idx)
    chans = ['chan_{}'.format(i) for i in idx]
    trueScaling = {chans[j]:pars[idx[j]] for j in range(len(chans))}

    # =========================
    # depictive
    # =========================

    print('===========================')
    print('Depictive Inference')
    dpi = inference(data, chans, 'fractionAlive', 's', 'a.u.')

    print('goodness of fit')
    print('rsq : {:0.3f}'.format(dpi.get_rsq()))

    print('===========================')
    print('Depictive Results')

    print('Name\tVarExplained')
    for name in dpi.get_chan_names():
        print('{}\t{:0.3f}'.format(name, dpi.get_var_explained(name)))

    print('Name\tTrueScaling\tInferScaling')
    for name in dpi.get_chan_names():
        print('{}\t{:0.3f}\t{:0.3f}'.format(name,
                trueScaling[name],
                dpi.get_scaling(name)))

    if not os.path.exists('figs'):
        os.mkdir('figs')


    for name in dpi.get_target_names():
        sname = os.path.join('figs', name + '.pdf')
        plots.response(dpi, name, savename=sname)

    plots.coefficientDetermination(dpi,
            savename=os.path.join('figs', 'coefDetermination.pdf'))
    return 0

# ========================================
# RUN PROGRAM
# ========================================

if __name__ == '__main__':
    # =================================
    # parse input
    # =================================
    parser = argparse.ArgumentParser()
    parser.add_argument('--pars', dest='pars',
                        default=PARS, nargs='+',
                        type=float)
    parser.add_argument('--ndoses', dest='ndoses', default=NDOSES, type=int)
    parser.add_argument('--ncells', dest='ncells', default=NCELLS, type=int)
    parser.add_argument('--idx', dest='idx', default=IDX, nargs='+', type=int)

    args = parser.parse_args()

    main(args.pars, args.ndoses, args.ncells, args.idx)
