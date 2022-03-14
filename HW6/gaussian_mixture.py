# This file is for gaussian_mixture implementation for CS274A HW6
# Author: Yiyang Min

import numpy as np
from scipy.stats import multivariate_normal
import random
from k_means import k_means, estimate_label
from plot_gauss_parameters import plot_gauss_parameters, bivariate_normal
import matplotlib.pyplot as plt
import time


color_map = ['red', 'blue', 'green', 'black', 'yellow', 'cryan', 'black', 'purple', 'orange', 'pink']


class GaussianComponent(object):
    def __init__(self, weight=1, d=2, mean=None, covariance=None):
        self.d = d
        self.mean = mean if type(mean) != type(None) else np.zeros(d)
        self.covariance = covariance if type(covariance) != type(None) else np.eye(d)
        self.weight = weight

    def __str__(self):
        return f"weight {self.weight}, mean {self.mean}, \ncovariance {self.covariance}"

    def prob(self, x):
        if x.shape[0] != self.d:
            raise ValueError
        if self.d == 2:
            return self.weight * bivariate_normal(x[0], x[1], np.sqrt(self.covariance[0,0]),  np.sqrt(self.covariance[1,1]),
                                                  self.mean[0], self.mean[1], sigmaxy=self.covariance[0,1])
        else:
            return self.weight * multivariate_normal.pdf(x, self.mean, self.covariance)

    def plot(self, colorstr='r', ax=None, delta=.001):
        if not ax:
            ax = plt
        plot_gauss_parameters(self.mean, self.covariance, colorstr, delta, ax)

    def print(self):
        print(self.__str__())


def gaussian_mixture(data, K, init_method="random", epsilon=1e-6, niterations=300, plotflag=False, plot_title="", rseed=123, debugmode=False):
    """Run gaussian mixture model on data

    Args:
        data: N x d data matrix with float type data
        K: number of clusters. K < N
        init_method: initialization method
        epsilon: convergence threshold used to detect convergence
        niterations (optional): maximum number of iterations to perform (default 500)
        plotflag (optional): equals 1 to plot parameters during learning,
                             0 for no plotting (default is 0)
        plot_title (optional): server as suptitle for plt
        rseed (optional): initial seed value for the random number generator
        debugmode: if true, print out more information for debugging

    Returns:
        init_gparams: initial gparams
        gparams: K-dim array with each element to be a GaussianComponent object
        memberships: N x K matrix of probability memberships for "data". [i, j]th element
             represents probability of ith data generates by jth gaussian component
        ll: log-likelihood value for each iteration

    Raises:
        ValueError: if K < N
    """
    N, d = data.shape
    if K > N:
        raise ValueError(
            f"n_samples={N} should be >= n_clusters={K}."
        )

    if rseed:
        random.seed(rseed)

    text11 = ""  # text for plot on ax[1, 1]

    # 1. Init
    init_gparams = init_gaussian_mixture(data, K, init_method, rseed)
    if debugmode:
        print(f"Initial parameter:")
    text11 += "Initial parameter:" + "\n"
    for i, gc in enumerate(init_gparams):
        if debugmode:
            print(f"{i+1}th gaussian ({color_map[i]}):")
            gc.print()
        text11 += f"{i+1}th gaussian ({color_map[i]}):" + "\n" + gc.__str__() + "\n"
    # plot init gparam
    if plotflag:
        fig, axs = plt.subplots(2, 2, figsize=(13, 13))
        ax = axs[0, 0]
        ax.set_title(f"Initial gaussian parameters")
        plot_data_and_gaussian(data, init_gparams, ax)
    gparams = np.array([GaussianComponent(gp.weight, gp.d, gp.mean) for gp in init_gparams])
    memberships = np.zeros((N, K))  # No need for exact initialization as the first E step would work

    # Variables for checking convergence
    ll = np.zeros(niterations + 2)
    ll[0] = -np.inf  # set as 0th iteration ll
    iteration = 0
    while True:
        iteration += 1
        # 2. E-step
        memberships, ll[iteration] = calculate_memberships_and_log_likelihood(data, gparams)

        # 3. M-step
        update_gparams(data, gparams, memberships)

        # 4. Compute log-lieklihood, done in step 2

        # 5. Check for convergence
        if iteration > niterations:
            break
        if np.absolute(ll[iteration] - ll[iteration-1]) < epsilon:
            break

        # Debug use: plot every 10 iteration
        if debugmode and iteration % 10 == 0:
            print(f"{iteration}th iteration debug -----")
            for i, gc in enumerate(gparams):
                print(f"{i}th gc:")
                gc.print()
            if plotflag:
                fig, ax = plt.subplots()
                plot_data_and_gaussian(data, gparams, ax)
                plt.show()
                time.sleep(3)  # wait for plotting

    # only return the non-zero ll
    if 0 in ll:
        zero_index = np.where(ll == 0)[0][0]
    else:
        zero_index = len(ll) + 1

    if debugmode:
        print(f"Final gaussian parameters")
        text11 += "\n" + "Final gaussian parameters:" + "\n"
    for i, gc in enumerate(gparams):
        if debugmode:
            print(f"{i+1}th gaussian ({color_map[i]}):")
            gc.print()
        text11 += f"{i+1}th gaussian ({color_map[i]}):" + "\n" + gc.__str__() + "\n"
    if plotflag:
        # plot gaussian mixture
        ax = axs[0, 1]
        ax.set_title(f"Initial gaussian parameters")
        plot_data_and_gaussian(data, gparams, ax)
        # plot log-likelihood
        ax = axs[1, 0]
        ax.set_title(f"Log-Likelihood by Iteration, Initialization: {init_method}")
        ax.plot(ll[1:zero_index])
        ax.set_xlabel("Iteration number")
        ax.set_ylabel("Log-Likelihood")
        # Plot text
        ax = axs[1, 1]
        ax.text(0, 0, text11)
        ax.axis("off")
        plt.suptitle(plot_title, fontsize="xx-large")
        plt.show()
    return gparams, memberships, ll[:zero_index]


def init_gaussian_mixture(data, K, init_method="random", rseed=123):
    """Init K-dim GaussianComponent array"""
    N, d = data.shape
    if init_method == "kmeans":
        means, labels, _ = k_means(data, k, rseed=rseed)
        tmp_cnt = [0 for _ in range(K)]
        for i, label in enumerate(labels):
            tmp_cnt[label] += 1
        gparams = np.array([GaussianComponent(tmp_cnt[i] / N, d, means[i]) for i in range(K)])
    else:  # "random" select K starting points
        # Random pick K points as center, assign label to each point based on distance and
        # hence update the probability
        random_list = list(range(N))
        random.shuffle(random_list)
        random_index = random_list[:K]
        means = [data[i] for i in random_index]
        labels = np.array([estimate_label(data[i], means) for i in range(N)])
        tmp_cnt = [0 for _ in range(K)]
        for i, label in enumerate(labels):
            tmp_cnt[label] += 1
        gparams = np.array([GaussianComponent(1 / K, d, means[i]) for i in range(K)])
    return gparams


def calculate_memberships_and_log_likelihood(data, gparams):
    """Calculate membership matrix"""
    N, d = data.shape
    K = len(gparams)
    memberships = np.zeros((N, K))
    log_likelihood = 0
    # Calculate probabilities first
    for i, x in enumerate(data):
        for j, gc in enumerate(gparams):  # gc - GaussianComponent
            memberships[i, j] = gc.prob(x)
    # Then calculate membership weights
    s = np.sum(memberships, axis=1)  # sum of each row
    log_likelihood += np.sum(np.log(s))
    memberships = np.divide(memberships, np.reshape(s, (N, 1)))
    # for i in range(N):
    #     s = sum(memberships[i])
    #     log_likelihood += np.log(s)
    #     for j in range(K):
    #         memberships[i, j] = memberships[i, j] / s
    return memberships, log_likelihood


def update_gparams(data, gparams, memberships):
    N, d = data.shape
    K = len(gparams)
    for j in range(K):
        # Update mean
        count = np.sum(memberships[:, j])
        mean = np.sum(data * memberships[:, [j]], axis=0)
        gparams[j].weight = count / N
        gparams[j].mean = np.reshape(mean, (d,)) / count
        # count = 0
        # mean = np.zeros(d)
        # # Update weight and mean
        # for i in range(N):
        #     count += memberships[i, j]
        #     mean += memberships[i, j] * data[i]
        # gparams[j].weight = count / N
        # gparams[j].mean = mean / count

        # Update covariance
        c_data = data - gparams[j].mean
        gparams[j].covariance = np.sum( np.reshape(memberships[:, j], (N, 1, 1)) *
                             np.reshape(c_data, (N, 1, d)) * np.reshape(c_data, (N, d, 1)), axis=0) / count
        # for i in range(N):
        #     covariance += memberships[i, j] * np.reshape(data[i] - gparams[j].mean, (1, d)) * np.reshape(data[i] - gparams[j].mean, (d, 1))
        # gparams[j].covariance = covariance / count
    return


def plot_data_and_gaussian(data, gparams, ax=None):
    if not ax:
        ax = plt
    ax.scatter(data[:, 0], data[:, 1], s=20, c='k', marker='x', alpha=.65, linewidths=2)
    for i, gc in enumerate(gparams):
        gc.plot(color_map[i], ax)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
