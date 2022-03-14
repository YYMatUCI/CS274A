# This file is for k_means implementation for CS274A HW6
# Author: Yiyang Min

import numpy as np
import random
import matplotlib.pyplot as plt


def k_means(data, K, plotflag=False, plot_title="", rseed=None):
    """Do K-means clustering over data.

    Args:
        data: N x d data matrix with float type data
        K: number of clusters. K < N
        plotflag (optional): equals 1 to plot parameters during learning,
                             0 for no plotting (default is 0)
        plot_title (optional): server as suptitle for plt
        rseed (optional): initial seed value for the random number generator

    Returns:
        means: K x d vector represents K d-dim mean vectors with lowest mean for this initialization.
        labels: label for each data point
        SSE: a float represents the SSE for this mean vector

    Raises:
        ValueError: if K < N
    """
    # 1. Initialize K mean vectors
    N, d = data.shape
    if K > N:
        raise ValueError(
            f"n_samples={N} should be >= n_clusters={K}."
        )

    if rseed:
        random.seed(rseed)

    random_list = list(range(N))
    random.shuffle(random_list)
    random_index = random_list[:K]
    means = np.array([data[i] for i in random_index])

    # Variables for checking convergence
    converge = False
    labels = [-1 for i in range(N)]  # label for each data point
    SSE = []  # calculated after each iteration
    while not converge:
        # 2. Estimate label for each data points
        for i in range(N):
            labels[i] = estimate_label(data[i], means)

        # 3. Compute new mean vectors
        new_means = calculate_mean(data, labels, K)
        SSE.append(calculate_SSE(data, labels, means))

        # 4. Check for convergence
        if np.array_equal(means, new_means):
            converge = True
        means = new_means

    if plotflag:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        # plot data
        plot_data_with_label(data, labels, means, axs[0])
        # plot SSE
        ax = axs[1]
        ax.plot(SSE)
        ax.set_title(f"Sum-Squared-Error by Iteration")
        ax.set_xlabel("Iteration number")
        ax.set_ylabel("Sum-Squared-Error")
        plt.suptitle(plot_title, fontsize="xx-large")
        plt.show()

    return means, labels, SSE


def estimate_label(point, means):
    """Return the closest mean id to point"""
    closest = -1
    dist = np.Inf
    for i, mean in enumerate(means):
        new_dist = np.linalg.norm(point - mean)
        if new_dist < dist:
            dist = new_dist
            closest = i
    return closest


def calculate_mean(data, labels, K):
    """Calculate mean with given labels"""
    N, d = data.shape
    tmp_sum = [np.zeros(d) for i in range(K)]
    tmp_cnt = [0 for i in range(K)]
    for i, label in enumerate(labels):
        tmp_sum[label] += data[i]
        tmp_cnt[label] += 1
    return np.array([tmp_sum[i] / tmp_cnt[i] for i in range(K)])


def calculate_SSE(data, labels, means):
    """Calculate sum-squared-error with given labels"""
    SSE = 0
    for i, label in enumerate(labels):
        SSE += np.inner(data[i] - means[label], data[i] - means[label])
    return SSE


def plot_data_with_label(data, label, mean, ax=None):
    if not ax:
        ax = plt
    scatter_data = ax.scatter(data[:, 0], data[:, 1], c=label)
    legend_data = ax.legend(*scatter_data.legend_elements(),  shadow=True, title="Cluster")
    ax.add_artist(legend_data)
    ax.scatter(mean[:, 0], mean[:, 1], c="orange", marker="x", s=100)
    ax.set_title(f"K-means clustering result, K = {len(mean)}")
