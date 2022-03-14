import numpy as np
import matplotlib.pyplot as plt


def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom


def plot_gauss_parameters(mu, covar, colorstr, delta=0.1, ax=None):

    """
    %PLOT_GAUSS:  plot_gauss_parameters(mu, covar,xaxis,yaxis,colorstr)
    %
    %  Python function to plot the covariance of a 2-dimensional Gaussian
    %  model as a "3-sigma" covariance ellipse  
    %
    %  INPUTS: 
    %   mu: the d-dimensional mean vector of a Gaussian model
    %   covar: d x d matrix: the d x d covariance matrix of a Gaussian model
    %   colorstr: string defining the color of the ellipse plotted (e.g., 'r')
    """
    # make grid
    x = np.arange(mu[0]-3.*np.sqrt(covar[0,0]), mu[0]+3.*np.sqrt(covar[0,0]), delta)
    y = np.arange(mu[1]-3.*np.sqrt(covar[1,1]), mu[1]+3.*np.sqrt(covar[1,1]), delta)
    X, Y = np.meshgrid(x, y)

    # get pdf values
    Z = bivariate_normal(X, Y, np.sqrt(covar[0,0]),  np.sqrt(covar[1,1]), mu[0], mu[1], sigmaxy=covar[0,1])

    # P.contour(X, Y, Z, colors=colorstr, linewidths=4)
    if not ax:
        ax = plt
    ax.contour(X, Y, Z, colors=colorstr, linewidths=4)


if __name__ == "__main__":
    plot_gauss_parameters(3+np.zeros((2,)), .5*np.eye(2), 'r')
    # P.show()
    plt.show()
