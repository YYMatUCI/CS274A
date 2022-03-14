# CS 274A HW6
# Author: Yiyang Min

import numpy as np
import matplotlib.pyplot as plt
from k_means import *
from gaussian_mixture import *
from plot_gauss_parameters import plot_gauss_parameters
from automatic_gaussian_mixture import *


init_method="random"  # For gaussian init. ["random", "kmeans"]
automatic_k_selection_min = 1  # Automatically K selection
automatic_k_selection_max = 5
automatic_k_selection_print = True
rseed = 123  # 0 for random
plotflag = True
debugmode = False
true_K = [2, 3, 2]  # inferred from dataset description
total_dataset = len(true_K)


datasets = [np.recfromtxt(f"dataset{i}.txt") for i in range(1, total_dataset + 1)]

# # K-means
for i, data in enumerate(datasets):
    k_means(data, true_K[i], plotflag=plotflag, plot_title=f"Dataset {i+1} K-means model")

# Gaussian mixture
for i, data in enumerate(datasets):
    gaussian_mixture(data, true_K[i], plotflag=plotflag,
                     plot_title=f"Dataset {i+1} gaussian mixture model", rseed=rseed, debugmode=debugmode)

# Automatically K selection
for i, data in enumerate(datasets):
    print(f"Dataset{i + 1}")
    k = automatix_gaussian_mixture(datasets[i], k_min=automatic_k_selection_min,
                                   k_max=automatic_k_selection_max, print_all=automatic_k_selection_print)
    print(f"In range({automatic_k_selection_min}, {automatic_k_selection_max+1}), selected K = {k}")

