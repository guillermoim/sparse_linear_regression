import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regressions import *

def filter_list(the_list, mask):
    flist = list()
    for i in range(mask.size):
        if mask[i]:
            flist.append(the_list[i])
    return flist

def load(filename):
    # Load matrix
    print(f'Loading data from {filename}\n')
    data = np.load(filename)
    return data['names'], data['complexities'], data['matrix'], data['labels']


def get_data(filename, cmax):
    print(f'Loading data from {filename}\n')
    values = []
    with open(filename, "r") as f:
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


def plot_cov(X):
    df = pd.DataFrame(data=X)
    f = plt.figure(figsize=(19, 15))
    df.corr()
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)


if __name__ == "__main__":
    # Text format
    # filename = "matrices/blocks_clear-k_8-6blocks.io"
    filename = "matrices/blocks_clear/blocks_clear-k_8-5blocks.io"
    # filename = "matrices/blocks_on/blocks_on-k_8-7blocks.io"
    # filename = "matrices/blocks_on/blocks_on-k_8-8blocks.io"
    # filename = "matrices/gripper/gripper-k_8-10balls.io"
    binary_data = False

    # Binary format
    # filename = "matrices/blocks_on_7.npz"
    c_max = 5

    oms_interval = np.arange(2, 5)
    lasso_positive = False
    lasso_complexities = False
    lasso_nlambda = 20

    methods = ['oms', 'lasso', 'l0learn'] #,

    if binary_data:
        names, complexities, X, y = load(filename)
    else:
        X, y, names, complexities = get_data(filename=filename, cmax=c_max)
    n, p = np.shape(X)

    # This mask should be in the data
    mask = np.empty(p,dtype=bool)
    for i in range(p):
        if i == 1 or i == 30:
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

    # consider data incrementing the features
    for fi in range(len(nfs)):
        mask[perm[fi]] = True
        print(f'Considering {np.sum(mask)} features')
        mX = X[:, mask]
        fil_names = filter_list(names, mask)
        fil_complexities = filter_list(complexities, mask)
        fil_idx = filter_list(range(0, p), mask)
        # plot_cov(mX)

        if 'oms' in methods:

            print('----------\nOMS Method\n----------\n')
            do_OMS(mX, y, oms_interval, fil_idx, fil_complexities, fil_names)

        if 'lasso' in methods:

            print('Lasso Method\n----------\n')
            do_lasso(mX, y, lasso_nlambda, lasso_positive, lasso_complexities, fil_idx, fil_complexities, fil_names)

        if 'l0learn' in methods:

            print('L0Learn Method\n----------\n')
            do_l0learn(mX, y, fil_idx, fil_complexities, fil_names)