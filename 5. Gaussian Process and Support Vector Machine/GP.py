import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

# prior
BETA = 5


def get_data(path):
    """
    Read input data
    """
    X, Y = [], []
    with open(os.path.join(path, 'input.data'), 'r') as f:
        data = f.read().strip().split('\n')
        for datum in data:
            x, y = datum.split(' ')
            X.append(x)
            Y.append(y)
    return np.array(X).astype(np.float), np.array(Y).astype(np.float)


def get_test_data(N=1000):
    X_test = np.linspace(-60, 60, N)
    return X_test


def rational_quadratic_kernel_nm(Xn, Xm, sigma, length, alpha):
    """
    Compute the similarty between two data points
    """
    # rational quadratic kernel
    return sigma**2 * (1 + ((Xn - Xm)** 2) / (2 * alpha * length**2)) ** (-alpha)


def compute_kernel_matrix(XN, XM, param):
    """
    Compute the kernel vector or kernel matrix.
    param:
        0: sigma
        1: length
        2: alpha
    """
    N, M = len(XN), len(XM)
    K = np.empty((N, M))
    for n in range(N):
        for m in range(M):
            K[n, m] = rational_quadratic_kernel_nm(XN[n], XM[m], param[0], param[1], param[2])
    return K

def get_variance(param):
    """
    Compute C the covariance matrix between training data points.
    This formula reflects the two Gaussian sources of randomness,
    that associated with y(x) and that associated with noise.
    """
    # randomness associated with y(x)
    prior_cov = compute_kernel_matrix(X_train, X_train, param)
    # randomness associated with noise
    noise_cov = np.identity(len(X_train)) / BETA
    # directly adding two covariance matrics and return the result
    # The reason we can do this addition is because this two distributions
    # are independent.
    return prior_cov + noise_cov


def predictive_distribution(param):
    """
    Compute the mean vector and covariance matrix of predictive distribution.

    intermidiate variable: vector k, cov matrix C, scalar c
    output: mean and variance
    """
    X_test = get_test_data(N=100)
    # Evaluate the intermidiate varialbes: k, C, c
    k_vector = compute_kernel_matrix(X_train, X_test, param)
    C = get_variance(param)
    inv_C = np.linalg.inv(C)
    c = compute_kernel_matrix(X_test, X_test, param) + (1.0 / BETA)

    # Based on the formulas in textbook,
    # compute the mean and variance of predictive distribution.
    means = k_vector.T.dot(inv_C).dot(Y_train)
    covs = c - k_vector.T.dot(inv_C).dot(k_vector)

    return X_test, means, covs


def GP_log_likelihood(param):
    """
    Gaussian process log likelihood,
    which derived from the standard form of
    a multivariate Gaussian distribution
    """
    C = get_variance(param)
    inv_C = np.linalg.inv(C)

    # three terms in log likelihood
    first = np.log(np.linalg.det(C))
    second = Y_train.T.dot(inv_C).dot(Y_train)
    third = len(X_train) * np.log(2 * np.pi)

    return (first + second + third) / 2


def optimize_kernel_hyperparameter():
    """
    Optimize the hyperparameters in kernel function.
    """
    # The kernel used in this homework including three hyperparameters.
    # The optimization apporach is conjugate gradient.
    result = optimize.minimize(GP_log_likelihood, np.array([1.0, 1.0, 1.0]), method='CG')
    return result.x


def plot():
    """
    - Show all training data points.
    - Draw a line to represent mean of f in range [-60,60].
    - Mark the 95% confidence interval of f.(2 std)
    """
    # plot predictive distribution which hyperparameters of kernel are randomly picked
    param = np.array([1.0, 1.0, 1.0])
    X_test, means, covs = predictive_distribution(param)
    stds = np.sqrt(np.diag(covs))
    print(stds.shape)

    plt.subplot(1, 2, 1)
    plt.plot(X_test, means, color='black')
    plt.fill_between(X_test, means - 2 * stds, means + 2 * stds, color='red', alpha=0.3)
    plt.scatter(X_train, Y_train, color='blue', s=80)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((-60, 60, y1, y2))
    plt.title('[sigma length alpha]\n[%s %s %s]' % (*param, ))


    # plot predictive distribution which kernel with optimal hyperparameters
    param = optimize_kernel_hyperparameter()
    X_test, means, covs = predictive_distribution(param)
    stds = np.sqrt(np.diag(covs))

    plt.subplot(1, 2, 2)
    plt.plot(X_test, means, color='black')
    plt.fill_between(X_test, means - 2 * stds, means + 2 * stds, color='red', alpha=0.3)
    plt.scatter(X_train, Y_train, color='blue', s=80)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((-60, 60, y1, y2))
    plt.title('[sigma length alpha]\n[%.8s %.8s %.8s]' % (*param, ))

    plt.show()


if __name__ == '__main__':
    X_train, Y_train = get_data('.')
    print(X_train.shape)
    print(X_train.dtype)
    print(Y_train.shape)
    print(Y_train.dtype)
    print(X_train)
    print(Y_train)
    plot()
    # optimize_kernel_hyperparameter()
