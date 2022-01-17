from json.tool import main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from regressions import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load(filepath):
    # Load matrix
    print(f'Loading data from {filepath}\n')
    data = np.load(filepath)
    return data['names'], data['complexities'], data['matrix'], data['labels']

def plot_cov(X, names, title='default'):
    
    df = pd.DataFrame(data=X)
    data = df.corr().to_numpy()
    data = np.tril(data)

    splits = [0, 40, 80, 120, 160, 200, 220]

    chunk = 0
    max_cols=1

    for row, idxX in enumerate(range(1, len(splits))):
        max_cols+=1
        for col, idxY in enumerate(range(1, max_cols)):

            print(f'plotting {row}, {col}')

            plt.close('all')

            f = plt.figure(figsize=(19, 15))

            #data in this chunk
            x_init, x_end = splits[idxX-1], splits[idxX]
            y_init, y_end = splits[idxY-1], splits[idxY]

            X = data[x_init:x_end,  y_init:y_end]

            plt.matshow(X, fignum=f.number)
            plt.xticks(range(X.shape[1]), names[y_init:y_end], fontsize=10, rotation=90)
            plt.yticks(range(X.shape[0]), names[x_init:x_end], fontsize=10)

            for (i, j), z in np.ndenumerate(X):
                if x_init+i >= y_init+j:
                    plt.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=14)
            plt.title(f'{title} CHUNK {chunk}', fontsize=16)
            plt.savefig(f'corr_plots/blocksworld_on/blocksworld_on_corr_CHUNK_{row}_{col}.pdf', bbox_inches='tight', dpi=500)
            chunk+=1


if __name__ == "__main__":
    
    filepath = "matrices/blocks_on_7.npz"
    c_max = 5

    names, complexities, X, y = load(filepath)
    sel = np.where(complexities <= c_max)[0]
    
    print('Total features', sel.shape)
    
    names, complexities, X, y = names[sel], complexities[sel], X[:, sel], y
    n, p = np.shape(X)

    # For complexity c_max = 5, the feature pool size 220
    # If I set 40

    plot_cov(X, names, f'Blocksworld:on - correlation matrix - $c_\max={c_max}$')
    exit()

