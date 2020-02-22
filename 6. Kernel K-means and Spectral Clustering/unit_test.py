import numpy as np
import scipy.spatial.distance as distance
import time
from contextlib import contextmanager


CHECK = True

def check_spatial(spatial, print_num=105):
    assert isinstance(spatial, np.ndarray)
    assert spatial.shape == (10000, 2)
    for i, v in enumerate(spatial):
        if i == print_num:
            break
        print(v, end=' ')

    print('\nCheck spatial end.\n')


def test_speed_cdist_pdistSquare(array):
    start = time.time()
    a = distance.cdist(array, array, metric='sqeuclidean')
    print('cdist:', time.time() - start)
    start = time.time()
    a = distance.pdist(array, metric='sqeuclidean')
    distance.squareform(a)
    print('pdist:', time.time() - start)


def test_second_term(num_data, k, kernel, prev_data_cluster):
    # 2 secs
    with timeit('Fast second term'):
        temp_classification = np.empty((num_data, k))
        for num_n in range(num_data):
            for num_k in range(k):
                temp_classification[num_n, num_k] = np.sum(kernel[num_n, prev_data_cluster == num_k])
    with timeit('(USE)Fast 2 second term'):
        # for num_n in range(num_data):
        tep_classification = np.empty((num_data, k))
        for num_k in range(k):
            k_split = kernel[:, prev_data_cluster == num_k]
            tep_classification[:, num_k] = np.sum(k_split, axis=1)
    # 120 secs
    with timeit('naive'):
        tmp_classification = np.empty((num_data, k))
        for i in range(num_data):
            for j in range(k):
                result = 0
                for m in range(num_data):
                    if prev_data_cluster[m] == j:
                        result += kernel[i][m]
                tmp_classification[i][j] = result
    print('difference second term:', np.sum(tmp_classification - temp_classification))
    print('difference second term2:', np.sum(tmp_classification - tep_classification))


@contextmanager
def timeit(name=''):
    start_time = time.time()
    yield
    elapse_time = time.time() - start_time

    print(name, '{}'.format(elapse_time))
