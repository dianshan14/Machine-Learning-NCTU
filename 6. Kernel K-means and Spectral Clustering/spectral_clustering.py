import os
import time
from sys import exit

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

import unit_test
from common import set_argument, get_data
from common import get_kernel, plot


def init_mean(k, init_method, k_space_data=None):
    print(init_method)
    if init_method == 'uniform_distribution':
        k_means = np.random.rand(k, k)
    elif init_method == 'random_pick_from_data':
        pick = np.random.randint(k_space_data.shape[0], size=k)
        k_means = k_space_data[pick]
    elif init_method == 'random_assignment':
        data_cluster = np.random.randint(k, size=k_space_data.shape[0])
        k_means = update_center(k_space_data, data_cluster)
    elif init_method == 'k_means++':
        mean_eigen_space = np.zeros((k, data_eigen_space.shape[1]), dtype=float)
        probability = np.ones((data_eigen_space.shape[0]), dtype=float) / data_spatial.shape[0]
        for cluster_idx in range(k):
            data_idx = np.random.choice(data_eigen_space.shape[0], size=1, p=probability)[0]
            mean_eigen_space[cluster_idx] = data_eigen_space[data_idx]
            if cluster_idx != k-1:
                probability = np.linalg.norm(data_eigen_space - mean_eigen_space[cluster_idx], axis=1)
                probability /= np.sum(probability)
        classification = classify(data_eigen_space, mean_eigen_space, k)
    else:
        raise NotImplementedError('INIT_METHOD: %s not defined' % init_method)

    print('init_mean', k_means.shape)
    return k_means


def clustering(k_space_data, k_means):
    # implement k-meas assignment
    distance_to_mean = distance.cdist(k_space_data, k_means, metric='sqeuclidean')
    return np.argmin(distance_to_mean, axis=1)


def update_center(k_space_data, data_cluster):
    N, k = k_space_data.shape

    values, counts = np.unique(data_cluster, return_counts=True)
    for i in range(k):
        if i not in values:
            counts = np.insert(counts, i, 1)
    assert counts.shape[0] == k, 'Different clusters: %s (%s)' % (counts.shape[0], values)

    k_means = np.empty((k, k))
    for num_k in range(k):
        k_means[num_k] = np.sum(k_space_data[data_cluster == num_k, :], axis=0) / counts[num_k]
    return k_means


def kmeans_cluster(k, k_space_data, init_method):
    result = []
    k_means = init_mean(k, init_method, k_space_data)
    data_cluster = clustering(k_space_data, k_means)

    result.append(data_cluster.reshape(args.img_size, args.img_size))

    it, converge = 1, 0
    prev_error = 0
    while(True):
        with unit_test.timeit('Time: '):
            k_means = update_center(k_space_data, data_cluster)
            prev_data_cluster = data_cluster
            data_cluster = clustering(k_space_data, k_means)

            result.append(data_cluster.reshape(args.img_size, args.img_size))

            error = np.sum(np.abs(data_cluster - prev_data_cluster))
            print('Iteration %s:' % it, error)
            if np.abs(error - prev_error) < 3 and it > 30:
                converge += 1
                if converge >= 3:
                    break
            else:
                converge = 0

            prev_error = error
            it += 1

    plot_setting = {}
    plot_setting['method'] = 'spectral_clustering (%s)' % args.cut_type
    plot(result, k, args, plot_setting)
    return data_cluster


def plot_eigen_space(k, k_space_data, data_cluster):
    color_mapping = ['red', 'blue', 'green', 'yellow', 'black', 'gray']
    for i in range(k):
        plt.scatter(k_space_data[data_cluster == i, 0:1],
                    k_space_data[data_cluster == i, 1:2],
                    s=8, c=color_mapping[i])

    TEXT = [
        'spectral_clustering (%s)  INIT_METHOD: %s' % (args.cut_type, args.init_method),
        'k=%s, gamma_s=%.3f, gamma_c=%.3f' % (args.k, args.gamma_s, args.gamma_c),
    ]
    plt.title('\n'.join(TEXT))
    plt.savefig("%s/plot_%s_%s.png" % (args.PATH, args.image, k))


def spectral_clustering(W, k, init_method_for_kmeans, cut_type):
    # get graph laplacian
    with unit_test.timeit('Laplacian'):
        # construct degree matrix
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        if cut_type == 'normalized_cut':
            norm_D = np.linalg.inv(np.power(D, 1 / 2))
            L = np.matmul(norm_D, L, norm_D)

    # get eigenvectors
    with unit_test.timeit('Eigen'):
        saved_vectors = 'vector_%s_%s_%s_%s.npy' % (args.image, args.gamma_s, args.gamma_c, args.img_size)
        saved_values = 'value_%s_%s_%s_%s.npy' % (args.image, args.gamma_s, args.gamma_c, args.img_size)
        if os.path.isfile(saved_values):
            eigen_values = np.load(saved_values)
            eigen_vectors = np.load(saved_vectors)
            print('load eigen')
        else:
            eigen_values, eigen_vectors = np.linalg.eig(L)
            np.save(saved_vectors, eigen_vectors)
            np.save(saved_values, eigen_values)
            print('save eigen')

    # construct U(N, k), which is k-dimensional
    # Euclidean space for each data points
    with unit_test.timeit('K eigen'):
        # sort eigenvectors by eigenvalues
        k_index = np.argsort(eigen_values)
        eigen_vectors = eigen_vectors[:, k_index]

        # k vectors after Fiedler vector
        U = eigen_vectors[:, 1:k+1]

    # kmeans on k-d Euclidean space
    with unit_test.timeit('K means'):
        data_cluster = kmeans_cluster(k, U, init_method_for_kmeans)

    plot_eigen_space(k, U, data_cluster)


if __name__ == '__main__':
    args = set_argument()
    args.img_size, color, spatial = get_data(filename='image%s.png' % args.image, scaledown=True)
    # args.img_size, color, spatial = get_data(filename='image%s.png' % args.image)
    spatial = np.true_divide(spatial, args.img_size-1)
    print('color', color.shape)
    print('spatial', spatial.shape)

    saved_kernel_filename = 'kernel_%s_%s_%s_%s.npy' % (args.image, args.gamma_s, args.gamma_c, args.img_size)
    if os.path.isfile(saved_kernel_filename):
        kernel = np.load(saved_kernel_filename)
        print('load kernel')
    else:
        with unit_test.timeit('Compute kernel time'):
            kernel = get_kernel(spatial, color, gamma_s=args.gamma_s,
                                    gamma_c=args.gamma_c)
        np.save(saved_kernel_filename, kernel)
        print('save kernel')

    args.init_method = 'uniform_distribution'
    args.init_method = 'random_pick_from_data'
    args.init_method = 'random_assignment'

    args.PATH = 'report/' + '%s_%s_%s_%s_%s_%s_100' % (args.image, args.k, args.init_method,
                                                   args.gamma_s, args.gamma_c, args.cut_type)
    os.system('mkdir -p %s' % args.PATH)
    spectral_clustering(W=kernel, k=args.k, init_method_for_kmeans=args.init_method, cut_type=args.cut_type)
