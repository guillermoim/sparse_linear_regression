import scipy
import numpy as np

from .score import score

from glmnet_python import glmnet
from glmnet_python.glmnetCoef import glmnetCoef
from glmnet_python.glmnetPredict import glmnetPredict
from glmnet_python.glmnetPlot import glmnetPlot

def do_lasso(X, true_target, noisy_target, noise_level, lasso_nlambda, lasso_force_positive, lasso_penalty_complexities, idx, complexities, names):
    # Check https://github.com/bbalasub1/glmnet_python/blob/master/test/glmnet_examples.ipynb
    # https://martin-becker.net/blog/error-running-pythons-glmnet-implementation-glmnet_py-missing-libgfortran-so-3/
    n, p = np.shape(X)
    xf = scipy.float64(X)
    yf = scipy.float64(noisy_target)
    y2 = scipy.random.rand(np.size(yf, 0), 1)
    for ind in range(noisy_target.shape[0]):
        y2[ind] = yf[ind]
    cl = scipy.array([[0.0], [50]], dtype=scipy.float64)
    pfac = scipy.zeros([1, p])
    for i in range(p):
        pfac[0, i] = complexities[i] - 1
    #print(pfac)
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
    
    res = []

    for l in range(len(vlambda)):

        feat_names = []
        feat_weights = []
        feat_complexities = []

        print(f"Lambda={vlambda[l]}")
        yp = np.squeeze(np.array(fc[:, l]))
        for i in range(p):
            if nc[i, l]:
                print(f"\tFeat {idx[i]}\t(c={complexities[i]})\tweight {coef[i + 1, l]:.14}:\t{names[i]}")
                feat_weights.append(np.round(coef[i + 1, l], 4))
                feat_complexities.append(complexities[i])
                feat_names.append(names[i])

        print(f"Score {score(yt, yp)}")

        entry = {"method": "lasso", "parameter": vlambda[l], "total_features":p, "noise_level": noise_level, 
                 "noisy_score": score(yp, noisy_target), "score": score(yp, true_target), 
                 "features_names": feat_names, "feature_weights" : feat_weights, "feature_complexities":feat_complexities}

        res.append(entry)

    return res



