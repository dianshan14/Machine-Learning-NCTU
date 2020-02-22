import numpy as np
import os
import csv
import time
import scipy.optimize as optimize
import scipy.spatial.distance as distance

from libsvm.python.svmutil import svm_predict, svm_read_problem, svm_train, svm_parameter, svm_problem


def get_data(path):
    files = ['X_train.csv', 'Y_train.csv', 'X_test.csv', 'Y_test.csv']
    data = []
    for filename in files:
        npy_filename = filename.split('.')[0] + '.npy'
        if os.path.isfile(npy_filename):
            data.append(np.load(npy_filename))
        else:
            with open(os.path.join(path, filename), 'r', newline='') as f:
                datum = np.array(list(csv.reader(f))).astype(np.float32)
                data.append(datum)
                np.save(npy_filename, datum)

    return (*data, )


def train_svm(kernel, parameter_for_training=''):
    """
    Train SVM with specific kernel (linear, polynomial, or RBF)
    """
    kernel2int = {'linear': 0, 'polynomial': 1, 'RBF': 2}

    start = time.time()
    # set training arguments and training
    model = svm_train(Y_train, X_train,
                      parameter_for_training + ' -q -s 0 -t %d' % (kernel2int[kernel]))
    # prediction
    pred_label, pred_acc, pred_val = svm_predict(Y_test, X_test, model)
    return pred_acc[0], time.time() - start


def search(kernel):
    """
    Grid Search:
        linear: C(cost)
        polynomial: C, gamma, degree, coef0
        RBF: C, gamma
    """
    print('Start searching...')
    with open('%s_search1.txt' % kernel, 'w') as f:
        # set up the search space
        cost_list = [0.1, 1, 10]
        gamma_list = [1e-3, 1e-6, 1e-9]
        degree_list = [3, 5, 10]
        coef0_list = [0, 1]
        # logging the search detail to file
        print('Cost: %s\ngamma: %s\ndegree: %s\ncoef0: %s'
              % (cost_list, gamma_list, degree_list, coef0_list), file=f)
        if kernel == 'linear':
            # grid search for linear kernel (cost)
            for cost in cost_list:
                try:
                    acc, elapsed_time = train_svm(kernel, '-c %.14f' % (cost, ))
                    print('%f,%f,%f' % (cost, acc, elapsed_time), file=f)
                    print([cost, acc, elapsed_time])
                except:
                    print('%f(FAIL)' % (cost, ), file=f)
                f.flush()
        elif kernel == 'polynomial':
            # grid search for polynomial kernel (cost, gamma, degree, coef0)
            for cost in cost_list:
                for gamma in gamma_list:
                    for degree in degree_list:
                        for coef0 in coef0_list:
                            try:
                                acc, elapsed_time = train_svm(kernel,
                                                              '-c %.14f -g %.14f -d %d -r %.8f'
                                                              % (cost, gamma, degree, coef0))
                                print('%f,%f,%f,%f,%f,%f'
                                      % (cost, gamma, degree, coef0, acc, elapsed_time), file=f)
                                print([cost, gamma, degree, coef0, acc, elapsed_time])
                            except:
                                print('%f,%f,%f,%f(FAIL)'
                                      % (cost, gamma, degree, coef0), file=f)
                            f.flush()
        elif kernel == 'RBF':
            # grid search for RBF kernel (cost, gamma)
            for cost in cost_list:
                for gamma in gamma_list:
                    try:
                        acc, elapsed_time = train_svm(kernel, '-c %.14f -g %.14f'
                                                      % (cost, gamma))
                        print('%f,%f,%f,%f'
                              % (cost, gamma, acc, elapsed_time), file=f)
                        print([cost, gamma, acc, elapsed_time])
                    except:
                        print('%f,%f(FAIL)'
                              % (cost, gamma), file=f)
                    f.flush()
        else:
            assert False


def linear_RBF_kernel_train(gamma=-0.25):
    """
    Computing the precomputed kernel (linear+RBF) for LIBSVM

    Based on the definition of radial basis function kernel from wiki
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    """
    linear_kernel = np.dot(X_train, X_train.T)

    # official document for distance.pdist:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    # Computes the squared Euclidean distance
    RBF_kernel_condensed = np.exp(-gamma * distance.pdist(X_train, 'sqeuclidean'))
    # Converts condensed distance matrices to square distance matrices
    RBF_kernel = distance.squareform(RBF_kernel_condensed)


    # According to this stack overflow post
    # https://stackoverflow.com/questions/7715138/using-precomputed-kernels-with-libsvm
    # we must include sample serial number as the first column of the training and testing data
    train_kernel = np.hstack((np.arange(1, 5001)[:, None], linear_kernel + RBF_kernel))
    return train_kernel

def linear_RBF_kernel_test(gamma=-0.25):
    """
    Computing the precomputed kernel (linear+RBF) for LIBSVM

    Based on the definition of radial basis function kernel from wiki
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel
    """
    linear_kernel = np.dot(X_test, X_train.T)

    # official document for distance.cdist:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    RBF_kernel = np.exp(-gamma * distance.cdist(X_test, X_train, 'sqeuclidean'))

    # According to this stack overflow post
    # https://stackoverflow.com/questions/7715138/using-precomputed-kernels-with-libsvm
    # we must include sample serial number as the first column of the training and testing data
    test_kernel = np.hstack((np.arange(1, 2501)[:, None], linear_kernel + RBF_kernel))
    return test_kernel


def train_search_user_defined_kernel():
    """
    Funciton for training and applying grid search for linear+RBF kernel
    """
    print('Start searching user-defined linear+RBF kernel...')
    with open('%s_search1.txt' % 'udefined', 'w') as f:
        cost_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        gamma_list = [1e-3, 1e-6, 1e-9]
        print('Cost: %s\ngamma: %s\n'
              % (cost_list, gamma_list), file=f)
        for cost in cost_list:
            for gamma in gamma_list:
                try:
                    start = time.time()

                    prob  = svm_problem(Y_train, linear_RBF_kernel_train(gamma), isKernel=True)
                    param = svm_parameter('-q -s 0 -t 4 -c %.14f' % cost)
                    model = svm_train(prob, param)
                    pred_label, pred_acc, pred_val = svm_predict(Y_test, linear_RBF_kernel_test(gamma), model)

                    elapsed_time = time.time() - start
                    print('%f,%f,%f,%f'
                          % (cost, gamma, pred_acc[0], elapsed_time), file=f)
                    print([cost, gamma, pred_acc[0], elapsed_time])
                except:
                    print('%f,%f(FAIL)'
                          % (cost, gamma), file=f)
                f.flush()


if __name__ == '__main__':
    import time
    start = time.time()
    X_train, Y_train, X_test, Y_test = get_data('.')
    Y_train = Y_train.astype(np.int).squeeze()
    Y_test  = Y_test.astype(np.int).squeeze()
    print('X_train', X_train.shape)
    print('Y_train', Y_train.shape)
    print('X_test', X_test.shape)
    print('Y_test', Y_test.shape)

    # part 1
    # train_svm('linear')
    print(time.time() - start)
    # train_svm('polynomial')
    print(time.time() - start)
    # train_svm('RBF')
    print(time.time() - start)

    # part 2
    # search('linear')

    # part 3
    # train_search_user_defined_kernel()
