import numpy as np

from .score import score
from sklearn.linear_model import OrthogonalMatchingPursuit


def do_OMP(X, y, omp_interval, idx, complexities, names):
    # Check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
    n, p = np.shape(X)
    for s in omp_interval:
        if s <= p:
            print(f"find {s} best features out of {p} features")
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
            omp.fit(X, y)
            coef = omp.coef_
            idx_r, = coef.nonzero()
            for i in idx_r:
                print(f"\tFeat {idx[i]}\t(c={complexities[i]})\tweight {coef[i]:.2}:\t{names[i]}")
            print(f"Score {omp.score(X, y)}")
            #print(f"Score {score(y, omp.predict(X))}")
