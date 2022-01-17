import numpy as np
import os
from regression_clear import get_data, load

filepath = 'matrices/blocks_clear/blocks_clear-k_8-5blocks.io'
filepath = 'matrices/blocks_on_7.npz'
#features, V, names, complexities = get_data(filepath, 5)
names, complexities, data, labels = load(filepath)


for i, name in enumerate(names):
    print(f'Feature index {i}, feature name {name}, complexity {complexities[i]}')
