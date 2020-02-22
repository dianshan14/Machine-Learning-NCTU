import numpy as np
import matplotlib.pyplot as plt

def uni_gaussian_generator(mean, variance):
    normal = np.random.uniform(size=12).sum() - 6
    return normal * np.sqrt(variance) + mean


def poly_basis_generator(n, W, a):
    """
    W: (n, 1)
    """
    W = np.array(W).reshape(1, -1)
    x = np.random.uniform(low=-1.0, high=1.0)
    A = np.array([x ** i for i in range(n)])
    return x, W.dot(A)[0] + uni_gaussian_generator(0, a)

def f(x, param, var=None, a=0, multiplier=1):
    result, n = np.zeros_like(x), len(param)
    for i_x in range(x.shape[0]):
        for i in range(n):
            result[i_x] += param[i] * x[i_x] ** i
        if var is not None:
            X = np.array([[x[i_x] ** i for i in range(n)]])
            result[i_x] += multiplier * ((a) + X.dot(var.dot(X.T))[0][0])
    return result

def plot(xy, mean_variance, a, gt_a=1):
    # plt.rcParams['figure.figsize'] = (14.0, 14.0)
    xy = np.array(xy)
    x_min = np.min(xy, axis=0)[0] - 1
    x_max = np.max(xy, axis=0)[0] + 1
    x = np.linspace(x_min, x_max, 3001)

    plt.subplot(2, 2, 1)
    # print function
    y = f(x, mean_variance[0][0])
    plt.plot(x, y, color='black')
    # plt.fill_between(x, f(x, mean_variance)-1, f(x, mean_variance)+1, color='red')
    plt.plot(x, f(x, mean_variance[0][0]) - gt_a, color='red')
    plt.plot(x, f(x, mean_variance[0][0]) + gt_a, color='red')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -15, 25))
    # plt.xticks(np.arange(round(x_min), round(x_max) + 1, 1))
    # plt.yticks(np.arange(-15, 30, 10))
    plt.title('Ground truth')

    plt.subplot(2, 2, 2)
    # print function
    plt.plot(x, f(x, mean_variance[1][0]), color='black')
    # plt.fill_between(x, f(x, mean_variance)-1, f(x, mean_variance)+1, color='red')
    plt.plot(x, f(x, mean_variance[1][0], mean_variance[1][1], a=a), color='red')
    plt.plot(x, f(x, mean_variance[1][0], mean_variance[1][1], a=a, multiplier=-1), color='red')
    # print points
    plt.scatter(xy[:, 0], xy[:, 1], color='blue')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -15, 25))
    plt.title('Predict result')

    plt.subplot(2, 2, 3)
    # print function
    plt.plot(x, f(x, mean_variance[2][0]), color='black')
    plt.fill_between(x, f(x, mean_variance[2][0], mean_variance[2][1], a=a),
                     f(x, mean_variance[2][0], mean_variance[2][1], a=a, multiplier=-1), color='pink')
    plt.plot(x, f(x, mean_variance[2][0], mean_variance[2][1], a=a), color='red')
    plt.plot(x, f(x, mean_variance[2][0], mean_variance[2][1], a=a, multiplier=-1), color='red')
    # print points
    plt.scatter(xy[:10, 0], xy[:10, 1], color='blue')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -15, 25))
    plt.title('After 10 incomes')

    plt.subplot(2, 2, 4)
    # print function
    plt.plot(x, f(x, mean_variance[3][0]), color='black')
    plt.fill_between(x, f(x, mean_variance[3][0], mean_variance[3][1], a=a),
                     f(x, mean_variance[3][0], mean_variance[3][1], a=a, multiplier=-1), color='pink')
    plt.plot(x, f(x, mean_variance[3][0], mean_variance[3][1], a=a), color='red')
    plt.plot(x, f(x, mean_variance[3][0], mean_variance[3][1], a=a, multiplier=-1), color='red')
    # print points
    plt.scatter(xy[:50, 0], xy[:50, 1], color='blue')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -15, 25))
    plt.title('After 50 incomes')

    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,x2,-100,100))

    plt.show()

def extract_standard_deviation(covariance_matrix):
    n = len(covariance_matrix)
    return np.sqrt(covariance_matrix[np.arange(n), np.arange(n)])

if __name__ == '__main__':
    for i in range(10):
        data = poly_basis_generator(5, [1,2,3,4,5], 1)
        print(data)
    mine = []
    for i in range(10000):
        mine.append(uni_gaussian_generator(3, 5))

    mine = np.array(mine)
    nump = np.random.normal(3, 5, size=10000)
    print()
    print(np.max(mine), np.max(nump))
    print(np.min(mine), np.min(nump))
    print(np.mean(mine), np.mean(nump))
    print(np.var(mine), np.var(nump))
