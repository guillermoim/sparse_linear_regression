import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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