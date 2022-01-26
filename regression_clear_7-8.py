import os
import sys
import random

import matplotlib.pyplot as plt
import numpy as np
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


def plot_cov(X, names, title='default'):
    df = pd.DataFrame(data=X)
    f = plt.figure(figsize=(19, 15))
    X = df.corr()
    X = np.tril(X)
    plt.matshow(X, fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), names, fontsize=10)

    for (i, j), z in np.ndenumerate(df.corr()):
        if i >= j:
            plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title(title, fontsize=16)
    
    plt.savefig('corr_plots/blocksworld_clear/blocksworld_clear_corr.pdf', bbox_inches='tight', dpi=500)


if __name__ == "__main__":
    # Text format
    filepath = "matrices/blocks_clear/blocks_clear-k_8-7blocks.io"

    # TODO: Select seed for reproducibility
    np.random.seed(10)
    random.seed(10)

    # MAX COMPLEXITY FEATURES
    c_max = 5

    omp_interval = np.arange(1, 10)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    methods = ['omp', ]#'lasso', 'l0learn'] 
    X, y, names, complexities = get_data(filepath=filepath, cmax=c_max)
    n, p = np.shape(X)

    # n = number of states, p = number of features
    n, p = np.shape(X)

    
    true_target = 2 * X[:, 30] + X[:, 1]

    # This mask should be in the data
    mask = np.empty(p, dtype=bool)

    # This is the features indices that are linearly combined 
    # to produce the value function.
    base_features = [1, 30]


    for i in range(p):
        if i in base_features:
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
    
    # Create a permuation of the non ground truth features
    perm = np.random.permutation(nfs)

    filename = os.path.basename(filepath)
    filename = filename.split('.')[0]


    # edit the stdout, to write out in a more appropiate buffer
    old_stdout = sys.stdout

    range_ = len(nfs)

    simulations = []

    # This loop increases one feature at a time from the nfs pool
    # (by setting the mask to True)
    for fi in range(range_):

            for noise_level in (0, .25, .5, 1):

                noisy_target = true_target + np.random.normal(scale=noise_level, size = X.shape[0])

                mask[perm[fi]] = True
                N = np.sum(mask)
                print(f'Considering {N} features, noise level {noise_level}')
                mX = X[:, mask]
                
                fil_names = filter_list(names, mask)
                fil_complexities = filter_list(complexities, mask)
                fil_idx = filter_list(range(0, p), mask)
                
                if 'omp' in methods:
                    os.makedirs(f'outputs/{filename}/omp', exist_ok=True)
                    path = f'outputs/{filename}/omp'
                    with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                        sys.stdout = f
                        print(f'Considering {N} features')
                        print('----------\nomp Method\n----------\n')
                        sim1 = do_OMP(mX, true_target, noisy_target, noise_level, omp_interval, fil_idx, fil_complexities, fil_names)
                        simulations.extend(sim1)

                if 'lasso' in methods:
                    os.makedirs(f'outputs/{filename}/lasso', exist_ok=True)
                    path = f'outputs/{filename}/lasso'
                    with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                        sys.stdout = f
                        print(f'Considering {N} features')
                        print('Lasso Method\n----------\n')
                        sim2 = do_lasso(mX, true_target, noisy_target, noise_level, lasso_nlambda, lasso_positive, lasso_complexities, fil_idx, fil_complexities, fil_names)

                if 'l0learn' in methods:
                    os.makedirs(f'outputs/{filename}/l0learn', exist_ok=True)
                    path = f'outputs/{filename}/l0learn'
                    with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                        sys.stdout = f
                        print(f'Considering {N} features')
                        print('L0Learn Method\n----------\n')
                        sim3 = do_l0learn(mX, true_target, noisy_target, noise_level, y, fil_idx, fil_complexities, fil_names)
                        
                        
                sys.stdout = old_stdout
    
    df = pd.DataFrame.from_records(simulations)
    print(df)

    df.to_csv('runs/clear_7_8.csv')
