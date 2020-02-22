import numpy as np
import numba as nb

from utils import get_parsed_MNIST, get_MNIST_data

x_train, y_train, x_test, y_test = get_parsed_MNIST('data')
x_train[x_train < 128] = 0
x_train[x_train >= 128] = 1
print(x_train[0, :])
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

N = x_train.shape[0]
D = 28 * 28
K = 10
pi = np.full(K, 0.1, dtype=np.float64)
mu = np.random.rand(K, D).astype(np.float64)
Z = np.full((N, K), 0.1, dtype=np.float64)
prev_mu = np.zeros((K, D), dtype=np.float64)

def E_step(Z, mu, pi, N, K):
    # update Z(N, K)
    for n in range(N):
        Zk = np.zeros(10, dtype=np.float64)
        for k in range(K):
            mul_D = mu[k, :] * x_train[n, :] + (1 - mu[k, :]) * (1 - x_train[n, :])
            Zk[k] = pi[k] * np.multiply.reduce(mul_D)
        bottom_term = np.sum(Zk)
        if bottom_term == 0:
            bottom_term = 1
        for k in range(K):
            Z[n, k] = Zk[k] / bottom_term
    return Z

    """
    for n in range(N):
        Zk = np.zeros(10, dtype=np.float64)
        for k in range(K):
            mul = np.float64(1.0)
            for i in range(D):
                if x_train[n][i]:
                    mul *= mu[k][i]
                else:
                    mul *= (1 - mu[k][i])
            Zk[k] = pi[k] * mul
        bottom_term = np.sum(Zk)
        if bottom_term == 0:
            bottom_term = 1
        for k in range(K):
            Z[n][k] = Zk[k] / bottom_term
    """

def M_step(Z, N, K, D):
    # update mu, pi
    """
    index = np.arange(N)
    mu1 = np.random.rand(K, D).astype(np.float64)
    mu2 = np.random.rand(K, D).astype(np.float64)
    for k in range(K):
        Nk = np.sum(Z[:, k])
        print(Nk.shape)
        for j in range(D):
            mu[k][j]  = np.sum(x_train[index, j] * Z[index ,k]) / Nk
            # temp = 0
            # for n in range(N):
            #     temp += x_train[n][j] * Z[n][k]
            # mu1[k][j] = temp / Nk
            # print('mu', mu[k][j])
            # print('mu1', mu1[k][j])
        pi[k] = Nk / N
    """

    Nks = np.sum(Z, axis=0)
    Nks[Nks == 0] = 1
    for k in range(K):
        for j in range(D):
            matrix_sum = x_train[:, j].dot(Z[:, k])
            mu[k][j] = matrix_sum / Nks[k]
            # Nk = Nks[k]
            # if Nk == 0:
            #     Nk = 1
            # mu[k][j] = (matrix_sum / Nk)
    for k in range(K):
        pi[k] = Nks[k] / N

    # print((mu1-mu).sum())
    return mu, pi

def check_reasonable_pi(pi, mu, Z, cont_success):
    success = True
    for k in range(K):
        if pi[k] == 0:
            print('UPDATE FAIL')
            # fail
            success = False
            # reset
            cont_success = 0
            pi = np.full(K, 0.1, dtype=np.float64)
            print(mu[0][:5])
            mu = np.random.rand(K, D).astype(np.float64)
            print(mu[0][:5])
            Z = np.full((N, K), 0.1, dtype=np.float64)
            break
    if success:
        print('UPDATE SUCCESS')
        # success
        cont_success += 1
    return pi, mu, Z, cont_success

def print_imagination(mu, title, mapping):
    # print mu (K, D)
    mu_copy = mu.copy()
    mu_copy[mu_copy < 0.5] = 0
    mu_copy[mu_copy >= 0.5] = 1
    mu_copy = mu_copy.astype(np.int)
    for i in range(K):
        label = mapping[i]
        print('\n%s %s:\n%s' % (title, i, mu_copy[label, 0]), end='')
        for j in range(1, D):
            print('%s%s' % ('\n' if (j % 28) == 0 else ' ', mu_copy[label, j]), end='')
        print()

def print_labeled_imagination():
    """
        match the unsupervised learning result and print it
    """
    table = np.zeros((10, 10), dtype=np.int)
    cluster_mapping = np.full(10, -1, dtype=np.int)
    for n in range(N):
        Zk = np.zeros(10, dtype=np.float64)
        for k in range(K):
            mul_D = mu[k, :] * x_train[n, :] + (1 - mu[k, :]) * (1 - x_train[n, :])
            Zk[k] = pi[k] * np.multiply.reduce(mul_D)
        # table[truth][pred]
        table[y_train[n]][np.argmax(Zk)] += 1

    k_range = np.arange(K)
    for i in range(K):
        index, label = np.unravel_index(np.argmax(table), table.shape)
        # cluster_mapping[truth] = predict
        cluster_mapping[index] = label # deteminate mapping
        table[index, k_range] = -1
        table[k_range, label] = -1

    print('-'*100)
    print(cluster_mapping)
    print('-'*100)

    print_imagination(mu, 'labeled_class', cluster_mapping)

    return cluster_mapping

def print_confusion_matrix(mu, pi, cluster_mapping):
    error = 60000
    confusion_matrix = np.zeros((10,2,2), dtype=np.int)
    for n in range(N):
        Zk = np.zeros(10, dtype=np.float64)
        for k in range(K):
            mul_D = mu[k, :] * x_train[n, :] + (1 - mu[k, :]) * (1 - x_train[n, :])
            Zk[k] = pi[k] * np.multiply.reduce(mul_D)

        predict = np.argmax(Zk)

        # mapping back to i
        # e.g. [3, 4, 2, 0, 1, 5, 6, 9, 7, 8]
        # original predict: real number
        # original predict: index of above array
        for i in range (K):
            if cluster_mapping[i] == predict:
                predict = i
                break

        for k in range(K):
            if y_train[n] == k:
                if predict == k:
                    # TP
                    confusion_matrix[k][0][0] += 1
                else:
                    # FN
                    confusion_matrix[k][0][1] += 1
            else:
                if predict == k:
                    # FP
                    confusion_matrix[k][1][0] += 1
                else:
                    # TN
                    confusion_matrix[k][1][1] += 1

    for i in range(0, 10):
        print("\n---------------------------------------------------------------\n")
        print("Confusion Matrix {}: ".format(i))
        print("\t\tPredict number {}\t Predict not number {}".format(i, i))
        print("Is number {}\t\t{}\t\t\t{}".format(i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
        print("Isn't number {}\t\t{}\t\t\t{}\n".format(i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
        print("Sensitivity (Successfully predict number {})\t: {}".format(i, confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])))
        print("Specificity (Successfully predict not number {})\t: {}".format(i, confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])))

    for i in range(0, 10):
        error -= confusion_matrix[i][0][0]
    return error


# k = 0
# n = 0
# accumulate = mu[k, :] * x_train[n, :] + (1 - mu[k, :]) * (1 - x_train[n, :])
# print(mu[k])
# print(accumulate)
# print(accumulate.shape)

# import time
# for i in range(10):
#     start = time.time()
#     E_step()
#     print(time.time()-start)
it = 0
cont_success = 0
while(True):
    import time
    start = time.time()

    # EM algorithm
    Z = E_step(Z, mu, pi, N, K)
    mu, pi = M_step(Z, N, K, D)

    # check if pi are reasonable
    # caused by continuosly multiplying numbers which are less than 1
    pi, mu, Z, cont_success = check_reasonable_pi(pi, mu, Z, cont_success)

    # compute difference between mu and prev_mu
    difference = np.sum(np.abs(mu - prev_mu))

    # print_imagination(mu, 'class', [i for i in range(K)])

    it += 1
    print('\nNo. of Iteration: %d, Difference: %s\n' % (it, difference))
    print('Elapsed time:', time.time() - start)
    print('Sum of pi %s' % np.sum(pi))
    print('-'*100, '\n')

    # stop condition
    if cont_success >= 5 and difference < 20 and np.sum(pi) > 0.95:
        break

    prev_mu = mu.copy()
    # --------------------------- end of EM ----------------------------------------- #

cluster_mapping = print_labeled_imagination()
error = print_confusion_matrix(mu, pi, cluster_mapping)
# result statistics
print("\nTotal iteration to converge: %s" % (it))
print("Total error rate: %s " % (error/N))

import sys; sys.exit()
# print confusion matrix
print("Newton's method':\n")
print('w:\n', w, '\n')
print('Confusion Matrix:\n\t\tPredict cluster 1  Predict cluster 2')
gt1_cls2 = int(y_pred[y == 0].sum())
gt2_cls2 = int(y_pred[y == 1].sum())
print('In cluster 1\t\t%s\t\t%s' % (N - gt1_cls2, gt1_cls2))
print('In cluster 2\t\t%s\t\t%s\n' % (N - gt2_cls2, gt2_cls2))
print('Sensitivity (Successfully predict cluster 1): %s' % ((N - gt1_cls2) / N))
print('Specificity (Successfully predict cluster 2): %s' % (gt2_cls2 / N))
print('Update times: %s (norm <= 1e-3)' % it)

