import numpy as np
import numba as nb
from utils import get_parsed_MNIST
from tools import initialize

N = 60000
D = 28 * 28
K = 10

@nb.jit
def get_training_data():
    x_train, y_train, x_test, y_test = get_parsed_MNIST('data')
    x_train[x_train < 128] = 0
    x_train[x_train >= 128] = 1
    return x_train.T, y_train

@nb.jit
def E_step(x_train, mu, pi, Z):
    for n in range(N):
        Zk = np.zeros(K, dtype=np.float64)
        for k in range(K):
            multiplier = np.float64(1.0)
            for i in range(D):
                if x_train[i][n]:
                    multiplier *= mu[i][k]
                else:
                    multiplier *= (1 - mu[i][k])
            Zk[k] = pi[k][0] * multiplier
        temp2 = np.sum(Zk)
        if temp2 == 0:
            temp2 = 1
        for k in range(K):
            Z[k][n] = Zk[k] / temp2
    return Z

@nb.jit
def M_step(x_train, mu, pi, Z):
    Nks = np.sum(Z, axis=1)
    for j in range(D):
        for k in range(K):
            matrix_sum = np.dot(x_train[j], Z[k])
            # print(x_train[j].shape)
            # print(Z[k].shape)
            # print(matrix_sum.shape)
            Nk = Nks[k]
            if Nk == 0:
                Nk = 1
            mu[j][k] = (matrix_sum / Nk)
    for k in range(K):
        pi[k][0] = Nks[k] / 60000
    return mu, pi

@nb.jit
def condition_check(pi, mu, mu_prev, Z, condition):
    temp = 0
    for i in range(K):
        # some classes keep impossible
        if pi[i][0] == 0 :
            # fail condition
            condition = 0 # reset condition
            temp = 1 # fail flag
            temp1 = mu_prev
            pi, mu, temp2, Z = initialize()
            mu_prev = temp1
            break
    if temp == 0:
        # correct condition
        condition += 1

    # if that is fail condition -> reset the parameter
    # else keep the parameters and add 1 to condition
    return pi, mu, mu_prev, Z, condition # if condition >= 8, successfully learning

@nb.jit
def difference(mu, mu_prev):
    temp = 0
    # mu(D, K) here
    for i in range(D):
        for j in range(K):
            temp += abs(mu[i][j] - mu_prev[i][j])
    return temp

@nb.jit
def print_mu(mu):
    MU_new = mu.transpose()
    for i in range(K):
        print("\nclass: ", i)
        for j in range(D):
            if j % 28 == 0 and j != 0:
                print("")
            if MU_new[i][j] >= 0.5:
                print("1", end=" ")
            else:
                print("0", end=" ")
        print("")

@nb.jit
def decide_label(x_train, y_train, mu, pi):
    table = np.zeros(shape=(10, 10), dtype=np.int)
    relation = np.full((10), -1, dtype=np.int)
    for n in range(0, 60000):
        temp = np.zeros(shape=10, dtype=np.float64)
        for k in range(0, 10):
            temp1 = np.float64(1.0)
            for i in range(0, 28 * 28):
                if x_train[n][i] == 1:
                    temp1 *= mu[i][k]
                else:
                    temp1 *= (1 - mu[i][k])
            temp[k] = pi[k][0] * temp1
        # table[truth][pred]
        table[y_train[n]][np.argmax(temp)] += 1
    print(table)
    print('-'*50, 'before', '-'*50)

    for i in range(1, 11):
        ind = np.unravel_index(np.argmax(table, axis=None), table.shape)
        # relation[truth] = predict
        relation[ind[0]] = ind[1]
        """
        for j in range(0, 10):
            # table[truth][j]
            table[ind[0]][j] = -1 * i
            # table[j][pred]
            table[j][ind[1]] = -1 * i
        """
        table[ind[0], :] = -1
        table[:, ind[1]] = -1
        print(table)

    return relation

@nb.jit
def print_labeled_class(mu, relation):
    MU_new = mu.transpose()
    for i in range(0, 10):
        print("\nlabeled class: ", i)
        label = relation[i]
        for j in range(0, 28 * 28):
            if j % 28 == 0 and j != 0:
                print("")
            if MU_new[label][j] >= 0.5:
                print("1", end=" ")
            else:
                print("0", end=" ")
        print("")

@nb.jit
def print_confusion_matrix(x_train, y_train, mu, pi, relation):
    error = 60000
    confusion_matrix = np.zeros(shape=(10,2,2), dtype=np.int)
    for n in range(0, 60000):
        temp = np.zeros(shape=10, dtype=np.float64)
        for k in range(0, 10):
            temp1 = np.float64(1.0)
            for i in range(0, 28 * 28):
                if x_train[n][i] == 1:
                    temp1 *= mu[i][k]
                else:
                    temp1 *= (1 - mu[i][k])
            temp[k] = pi[k][0] * temp1
        predict = np.argmax(temp)

        for i in range (0, 10):
            if relation[i] == predict:
                predict = i
                break

        for k in range(0, 10):
            if y_train[n] == k:
                if predict == k:
                    confusion_matrix[k][0][0] += 1
                else:
                    confusion_matrix[k][0][1] += 1
            else:
                if predict == k:
                    confusion_matrix[k][1][0] += 1
                else:
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

# start
x_train, y_train = get_training_data() # x_train = 784 * 60000
# pi, mu, mu_prev, Z = initialize() # pi = 10 * 1, mu = 784 * 10, Z = 10 * 60000
pi = np.full((10, 1), 0.1, dtype=np.float64)
mu = np.random.rand(28 * 28, 10).astype(np.float64)
mu_prev = np.zeros((28 * 28, 10), dtype=np.float64)
Z = np.full((10, 60000), 0.1, dtype=np.float64)

iteration = 0
condition = 0
import time
while(True):
    start = time.time()
    iteration += 1
    # E-step:
    Z = E_step(x_train, mu, pi, Z)

    # M-step:
    mu, pi = M_step(x_train, mu, pi, Z)

    # check pi
    pi, mu, mu_prev, Z, condition = condition_check(pi, mu, mu_prev, Z, condition)
    gap = difference(mu, mu_prev)
    if gap < 20 and condition >= 8 and np.sum(pi) > 0.95:
        break
    mu_prev = mu
    print_mu(mu)
    print("No. of Iteration: {}, Difference: {}\n".format(iteration, gap))
    print(time.time() - start)
    print("---------------------------------------------------------------\n")

print("---------------------------------------------------------------\n")
relation = decide_label(x_train.transpose(), y_train, mu, pi)
print('-'*200)
print(relation)
print('-'*200)
print_labeled_class(mu, relation)
error = print_confusion_matrix(x_train.transpose(), y_train, mu, pi, relation)
print("\nTotal iteration to converge: {}".format(iteration))
print("Total error rate: {}".format(error/60000.0))
