import os
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance

import unit_test
from common import set_argument, get_data
from common import get_kernel, plot


def init(k, init_method, num_data, spatial, color):
    if init_method == 'random_assignment':
        data_cluster = np.random.randint(k, size=num_data)
    elif init_method == 'equal_split':
        data_cluster = np.zeros(num_data, dtype=np.int)
        split = num_data // k
        for i in range(k):
            data_cluster[i*split:(i+1)*split] = i
        data_cluster[-1] = k-1
    elif init_method == 'spatial_cloest_picked':
        pick = np.random.randint(num_data, size=k)
        center = spatial[pick]
        distance_to_center = distance.cdist(spatial, center, metric='sqeuclidean')
        data_cluster = np.argmin(distance_to_center, axis=1)
    elif init_method == 'color_cloest_picked':
        pick = np.random.randint(num_data, size=k)
        center = color[pick]
        distance_to_center = distance.cdist(color, center, metric='sqeuclidean')
        data_cluster = np.argmin(distance_to_center, axis=1)
    else:
        raise NotImplementedError('INIT_METHOD: %s not defined' % init_method)

    return data_cluster


def clustering(k, kernel, num_data, prev_data_cluster):
    """
    Implement the Kernel K-means formula, which in slide p.22
    """

    values, counts = np.unique(prev_data_cluster, return_counts=True)
    assert counts.shape[0] == k, 'Different clusters: %s (%s)' % (counts.shape[0], values)
    # sum all the data point which belong to specific cluster

    # first term
    first_term = np.diag(kernel)[:, None]

    # unit_test.test_second_term(num_data, k, kernel, prev_data_cluster)
    # second term
    second_term = np.empty((num_data, k))
    for num_k in range(k):
        k_split = kernel[:, prev_data_cluster == num_k]
        second_term[:, num_k] = -2 * np.sum(k_split, axis=1)
    second_term /= counts

    # third term
    third_term = np.empty(k)
    for num_k in range(k):
        mask = prev_data_cluster == num_k
        third_term[num_k] = np.sum(kernel[np.ix_(mask, mask)])
    third_term /= counts ** 2

    new_data_cluster = np.argmin(first_term + second_term + third_term, axis=1)
    return new_data_cluster


def kernel_k_means(k, num_data, kernel, init_method, spatial, color):
    result = []
    prev_data_cluster = init(k, init_method, num_data, spatial, color)
    result.append(prev_data_cluster.reshape(args.img_size, args.img_size))
    data_cluster = clustering(k, kernel, num_data, prev_data_cluster)
    error = np.sum(np.abs(data_cluster - prev_data_cluster))

    result.append(data_cluster.reshape(args.img_size, args.img_size))
    it, converge = 1, 0
    prev_error = 0
    while(True):
        with unit_test.timeit('Time: '):
            prev_data_cluster = data_cluster
            data_cluster = clustering(k, kernel, num_data, prev_data_cluster)
            result.append(data_cluster.reshape(args.img_size, args.img_size))

            error = np.sum(np.abs(data_cluster - prev_data_cluster))
            print('Iteration %s:' % it, error)
            if np.abs(error - prev_error) < 3:
                converge += 1
                if converge >= 3:
                    break
            else:
                converge = 0

            prev_error = error
            it += 1

    plot_setting = {}
    plot_setting['method'] = 'kernel k-means'
    plot(result, k, args, plot_setting)


if __name__ == '__main__':
    args = set_argument()
    args.img_size, color, spatial = get_data(filename='image%s.png' % args.image)
    # normalize the data
    spatial = np.true_divide(spatial, 99)
    print('color', color.shape)
    print('spatial', spatial.shape)
    # unit_test.check_spatial(spatial)

    # unit_test.test_speed_cdist_pdistSquare(color)
    """
    saved_kernel_filename = 'kernel_%s.npy' % args.image
    if os.path.isfile(saved_kernel_filename):
        kernel = np.load(saved_kernel_filename)
        print('load kernel')
    else:
        with unit_test.timeit('Compute kernel time'):
            kernel = compute_kernel(spatial, color, gamma_s=args.gamma_s,
                                    gamma_c=args.gamma_c)
        np.save(saved_kernel_filename, kernel)
        print('save kernel')
    """
    with unit_test.timeit('Compute kernel time'):
        kernel = get_kernel(spatial, color, gamma_s=args.gamma_s,
                                gamma_c=args.gamma_c)

    args.PATH = 'report/kkmeans/' + '%s_%s_%s_%s_%s' % (args.image, args.k, args.init_method,
                                                        args.gamma_s, args.gamma_c)
    os.system('mkdir -p %s' % args.PATH)
    kernel_k_means(k=args.k, num_data=len(color), kernel=kernel, init_method=args.init_method,
                   spatial=spatial, color=color)
