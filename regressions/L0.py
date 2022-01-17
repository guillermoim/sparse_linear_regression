import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
import numpy as np
from .score import score

def do_l0learn(X, y, idx, complexities, names):
    # Check https://cran.r-project.org/web/packages/L0Learn/vignettes/L0Learn-vignette.html
    # Check https://github.com/rpy2/rpy2 (https://rpy2.github.io/doc/v3.4.x/html/introduction.html#r-packages)
    n, p = np.shape(X)
    l0learn = rpackages.importr('L0Learn')
    #print(X.shape)
    #print('Converting features to R matrix')
    Xf = X.flatten('F')
    #print('\tflatten done')
    Xr = robjects.IntVector(Xf)
    #print('\tvector created')
    XR = robjects.r['matrix'](Xr, nrow=X.shape[0])
    #print('\tmatrix created')
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