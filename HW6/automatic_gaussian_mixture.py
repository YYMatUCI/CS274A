# This file is for gaussian_mixture implementation for CS274A HW6
# Author: Yiyang Min

import numpy as np
from gaussian_mixture import *


def automatix_gaussian_mixture(data, k_min=1, k_max= 5, init_method="random", print_all=True, rseed=123):
    N, d = data.shape
    k_candidate = [i for i in range(k_min, k_max+1)]
    BIC_scores = [0.0 for _ in k_candidate]
    lls = np.array([None for _ in k_candidate])
    for i, k in enumerate(k_candidate):
        _, _, lls[i] = gaussian_mixture(data, k, init_method, rseed=rseed)
        BIC_scores[i] = BIC(lls[i][-1], k * (1 + d + d*(d+1)//2), N)
        if print_all:
            print(f"K:{k}, ll:{lls[i][-1]:8.4f}, BIC: {BIC_scores[i]:8.4f}")
    return k_candidate[BIC_scores.index(max(BIC_scores))]


def BIC(ll, pk, N):
    return ll - np.log(N)*pk/2
