import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import OrthogonalMatchingPursuit

from glmnet_python import glmnet
from glmnet_python.glmnetCoef import glmnetCoef
from glmnet_python.glmnetPredict import glmnetPredict
from glmnet_python.glmnetPlot import glmnetPlot

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages


def filter_list(the_list, mask):
    flist = list()
    for i in range(mask.size):
        if mask[i]:
            flist.append(the_list[i])
    return flist


def score(y_true, y_pred):
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return 1-u/v


def load(filename):
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


def do_OMS(X, y, oms_interval, idx, complexities, names):
    # Check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
    n, p = np.shape(X)
    for s in oms_interval:
        if s <= p:
            print(f"find {s} best features out of {p} features")
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
            omp.fit(X, y)
            coef = omp.coef_
            idx_r, = coef.nonzero()
            for i in idx_r:
                print(f"Feat {idx[i]}\t(c={complexities[i]})\tweight {coef[i]:.2}:\t{names[i]}")
            print(f"Score {omp.score(X, y)}")
            print(f"Score {score(y, omp.predict(X))}")


def do_lasso(X, y, lasso_nlambda, lasso_force_positive, lasso_penalty_complexities, idx, complexities, names):
    # Check https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb
    # https://martin-becker.net/blog/error-running-pythons-glmnet-implementation-glmnet_py-missing-libgfortran-so-3/
    n, p = np.shape(X)
    xf = scipy.float64(X)
    yf = scipy.float64(y)
    y2 = scipy.random.rand(np.size(yf, 0), 1)
    for ind in range(y.shape[0]):
        y2[ind] = yf[ind]
    cl = scipy.array([[0.0], [50]], dtype=scipy.float64)
    pfac = scipy.zeros([1, p])
    for i in range(p):
        pfac[0, i] = complexities[i] - 1
    print(pfac)
    if not lasso_force_positive and not lasso_penalty_complexities:
        fit = glmnet(x=xf, y=y2, nlambda=lasso_nlambda)
    if lasso_force_positive and not lasso_penalty_complexities:
        fit = glmnet(x=xf, y=y2, cl=cl, nlambda=lasso_nlambda)
    if lasso_force_positive and lasso_penalty_complexities:
        fit = glmnet(x=xf, y=y2, cl=cl, penalty_factor=pfac, nlambda=lasso_nlambda)

    # glmnetPlot(fit, xvar = 'lambda', label = True);
    # glmnetPlot(fit, xvar = 'dev', label = True);
    # print(any(fit['lambdau'] == 0.5))
    vlambda = fit['lambdau']
    yt = np.squeeze(np.array(y2))
    coef = glmnetCoef(fit, s=scipy.float64(vlambda), exact=False)
    fc = glmnetPredict(fit, xf, ptype='response', s=scipy.float64(vlambda))
    nc = glmnetPredict(fit, ptype='nonzero', s=scipy.float64(vlambda))
    for l in range(len(vlambda)):
        print(f"Lambda={vlambda[l]}")
        yp = np.squeeze(np.array(fc[:, l]))
        for i in range(p):
            if nc[i, l]:
                print(f"\tFeat {idx[i]}\t(c={complexities[i]})\tweight {coef[i + 1, l]:.14}:\t{names[i]}")
        print(f"Score {score(yt, yp)}")


def do_l0learn(X, y, idx, complexities, names):
    # Check https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html
    # Check https://github.com/rpy2/rpy2 (https://rpy2.github.io/doc/v3.4.x/html/introduction.html#r-packages)
    n, p = np.shape(X)
    l0learn = rpackages.importr('L0Learn')
    print(X.shape)
    print('Converting features to R matrix')
    Xf = X.flatten('F')
    print('\tflatten done')
    Xr = robjects.IntVector(Xf)
    print('\tvector created')
    XR = robjects.r['matrix'](Xr, nrow=X.shape[0])
    print('\tmatrix created')
    yr = robjects.FloatVector(y)
    rlearn = robjects.r['L0Learn.fit']
    l0fit = rlearn(XR, yr, penalty="L0", maxSuppSize=p)
    rprint = robjects.r['print']
    res = rprint(l0fit)
    vlambda = res[0]
    #vgamma = res[1]
    #vsupp = res[2]
    yt = np.squeeze(np.array(y))
    for l in range(len(vlambda)):
        print(f"Lambda={vlambda[l]}")
        rfunc = robjects.r['coef']
        res = rfunc(l0fit, vlambda[l], gamma=0)
        weights = res.slots['x']
        vidx = res.slots['i']
        lnc = res.slots['x']
        print(f"\tIntercept\t{weights[0]}")
        for i in range(1, len(lnc)):
            print(
                f"\tFeat {idx[vidx[i]-1]}\t(c={complexities[vidx[i]-1]})\tweight {weights[i]:.4}:\t{names[vidx[i]-1]}")

        rfunc = robjects.r['predict']
        res = rfunc(l0fit, XR, vlambda[l], gamma=0)
        yp = np.squeeze(np.array(res.slots['x']))
        print(f"Score {score(yt, yp)}")


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