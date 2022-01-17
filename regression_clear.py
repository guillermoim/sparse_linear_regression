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
    
    plt.savefig('blocksworld_clear_corr.pdf', bbox_inches='tight', dpi=500)


if __name__ == "__main__":
    # Text format
    # filepath = "matrices/blocks_clear-k_8-6blocks.io"
    filepath = "matrices/blocks_clear/blocks_clear-k_8-5blocks.io"
    # filepath = "matrices/blocks_on/blocks_on-k_8-7blocks.io"
    # filepath = "matrices/blocks_on/blocks_on-k_8-8blocks.io"
    # filepath = "matrices/gripper/gripper-k_8-10balls.io"
    binary_data = False

    # TODO: Select seed for reproducibility
    np.random.seed(10)
    random.seed(10)

    # Binary format
    # filepath = "matrices/blocks_on_7.npz"
    c_max = 5

    omp_interval = np.arange(2, 10)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    methods = ['omp', 'lasso', 'l0learn'] 
    if binary_data:
        names, complexities, X, y = load(filepath)
    else:
        X, y, names, complexities = get_data(filepath=filepath, cmax=c_max)
    n, p = np.shape(X)


    plot_cov(X, names, f'Blocksworld:clear - correlation matrix - $c_\max={c_max}$')
    exit()

    # Compose the target
    variance = 1
    y = 2 * X[:, 30] + X[:, 1] #+ np.random.normal(scale=variance, size = X.shape[0])

    # This mask should be in the data
    mask = np.empty(p,dtype=bool)

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
    perm = np.random.permutation(nfs)

    filename = os.path.basename(filepath)
    filename = filename.split('.')[0]

    old_stdout = sys.stdout

    range_ = len(nfs)

    # consider data incrementing the features
    for fi in range(range_):
            mask[perm[fi]] = True
            N = np.sum(mask)
            print(f'Considering {N} features')
            mX = X[:, mask]
            fil_names = filter_list(names, mask)
            fil_complexities = filter_list(complexities, mask)
            fil_idx = filter_list(range(0, p), mask)
            # plot_cov(mX)
            if 'omp' in methods:
                os.makedirs(f'outputs/{filename}/omp', exist_ok=True)
                path = f'outputs/{filename}/omp'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('----------\nomp Method\n----------\n')
                    do_OMP(mX, y, omp_interval, fil_idx, fil_complexities, fil_names)

            if 'lasso' in methods:
                os.makedirs(f'outputs/{filename}/lasso', exist_ok=True)
                path = f'outputs/{filename}/lasso'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('Lasso Method\n----------\n')
                    do_lasso(mX, y, lasso_nlambda, lasso_positive, lasso_complexities, fil_idx, fil_complexities, fil_names)

            if 'l0learn' in methods:
                os.makedirs(f'outputs/{filename}/l0learn', exist_ok=True)
                path = f'outputs/{filename}/l0learn'
                with open(f'{path}/{np.sum(mask)}features.out', 'w') as f:
                    sys.stdout = f
                    print(f'Considering {N} features')
                    print('L0Learn Method\n----------\n')
                    do_l0learn(mX, y, fil_idx, fil_complexities, fil_names)
            
            sys.stdout = old_stdout
        
    plot_cov(mX, fil_names, 'Blockworls:clear - correlation matrix - $c_\max=5$')
    #plt.show()