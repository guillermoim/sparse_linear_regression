from pyexpat import features
import numpy as np

from .score import score
from sklearn.linear_model import OrthogonalMatchingPursuit


def do_OMP(X, true_target, noisy_target, noise_level, omp_interval, idx, complexities, names):
    # Check https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html
    _, p = np.shape(X)

    res = []
    
    for s in omp_interval:
        if s <= p:
            
            print(f"find {s} best features out of {p} features")
            
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=s)
            omp.fit(X, noisy_target)
            coef = omp.coef_
            idx_r, = coef.nonzero()

            print('asasas', idx_r)
            
            feat_weights = []
            feat_complexities = []
            feat_names = []

            for i in idx_r:
                print(f"\tFeat {idx[i]}\t(c={complexities[i]})\tweight {coef[i]:.2}:\t{names[i]}")
                feat_weights.append(coef[i])
                feat_complexities.append(complexities[i])
                feat_names.append(names[i])

            print(f"Score {omp.score(X, true_target)}")
            
            entry = {"method": "omp", "parameter": s, "total_features":p, "noise_level": noise_level, 
                     "noisy_score": omp.score(X, noisy_target), "score": omp.score(X, true_target), 
                     "features_names": feat_names, "feature_weights" : feat_weights, "feature_complexities":feat_complexities}

            res.append(entry)

    return res
