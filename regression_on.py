import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.arrayprint import format_float_scientific
import pandas as pd

from regressions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def filter_list(the_list, mask):
    flist = list()
    for i in range(mask.size):
        if mask[i]:
            flist.append(the_list[i])
    return flist

def load(filepath):
    # Load matrix
    print(f'Loading data from {filepath}\n')
    data = np.load(filepath)
    return data['names'], data['complexities'], data['matrix'], data['labels']


def get_data(filepath, cmax):
    print(f'Loading data from {filepath}\n')
    values = []
    with open(filepath, "r") as f:
        # Line #0: comment line, simply ignore
        f.readline()

        # Line #1: feature names
        names = f.readline().rstrip().split(' ')

        # Line #2: feature complexities
        complexities = [int(x) for x in f.readline().rstrip().split(' ')]

        assert len(names) == len(complexities)

        # One line per state with the numeric denotation of all features
        for i, line in enumerate(f, start=0):
            values.append([int(x) for x in line.rstrip().split(' ')])

        ic = 0
        while cmax > complexities[ic]:
            ic = ic + 1

            # print(f"Read a matrix of {len(names)} features per {len(values)} states.")
        
        # print(f"Value of feature #7 in state #5 is {values[5][7]}")
        # print(f"hstar value of state #5 is {values[5][-1]}")

        #X = np.matrix(values)
        X = np.array(values)
        #y = np.squeeze(np.array(X[:,-1]))

        return X[:, 0:ic], X[:, -1], names[:ic], complexities[:ic]


if __name__ == "__main__":
    
    # Text format
    filepath = "matrices/blocks_on_7.npz"

    # TODO: Select seed for reproducibility
    np.random.seed(10)
    random.seed(10)

    # Binary format
    c_max = 8

    omp_interval = np.arange(2, 7)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    methods = ['oms', 'lasso', 'l0learn']

    names, complexities, X, y = load(filepath)
    sel = np.where(complexities <= c_max)[0]
    print('Total features', sel.shape)
    names, complexities, X, y = names[sel], complexities[sel], X[:, sel], y
    
    n, p = np.shape(X)

    target = np.copy(y)

    # This mask should be in the data
    mask = np.empty(p,dtype=bool)
    for i in range(p):
        if i < 500:
            mask[i] = True
        else:
            mask[i] = False

    # fs contains ground truth feats if provided, nfs contains the rest
    fs = []
    nfs = []
    for i in range(p):
        if mask[i]:
            fs.append(i)
        else:
            nfs.append(i)
    
    perm = np.random.permutation(nfs)

    filename = os.path.basename(filepath)
    filename = filename.split('.')[0]

    old_stdout = sys.stdout

    range_ = 450

    simulations = []


    # consider data incrementing the features
    for fi in range(range_):
         
         for noise_level in (0, .25, .5, 1):
            
            noisy_target = target + np.random.normal(scale=noise_level, size = X.shape[0])

            mask[perm[fi]] = True
            N = np.sum(mask)
            
            print(f'Considering {N} features')
            mX = X[:, mask]
            fil_names = filter_list(names, mask)
            fil_complexities = filter_list(complexities, mask)
            fil_idx = filter_list(range(0, p), mask)
            # plot_cov(mX)
            if 'oms' in methods:
                os.makedirs(f'outputs/{filename}/oms', exist_ok=True)
                path = f'outputs/{filename}/oms'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('----------\nOMS Method\n----------\n')
                    sim1 = do_OMP(mX, target, noisy_target, noise_level, omp_interval, fil_idx, fil_complexities, fil_names)
                    simulations.extend(sim1)

            if 'lasso' in methods:
                os.makedirs(f'outputs/{filename}/lasso', exist_ok=True)
                path = f'outputs/{filename}/lasso'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('Lasso Method\n----------\n')
                    sim2 = do_lasso(mX, target, noisy_target, noise_level, lasso_nlambda, lasso_positive, lasso_complexities, fil_idx, fil_complexities, fil_names)
                    simulations.extend(sim2)

            if 'l0learn' in methods:
                os.makedirs(f'outputs/{filename}/l0learn', exist_ok=True)
                path = f'outputs/{filename}/l0learn'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('L0Learn Method\n----------\n')
                    sim3 = do_l0learn(mX, target, noisy_target, noise_level, fil_idx, fil_complexities, fil_names)
                    simulations.extend(sim3)

            sys.stdout = old_stdout

    df = pd.DataFrame.from_records(simulations)
    print(df)

    df.to_csv('runs/blocksworld_on.csv', index=False)