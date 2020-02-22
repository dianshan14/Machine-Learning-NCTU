import matplotlib.pyplot as plt
import numpy as np

def f(x, param):
    result, n = np.zeros_like(x), len(param)
    for i_x in range(x.shape[0]):
        for i in range(n):
            result[i_x] += param[i][0] * x[i_x] ** (n-i-1)
    return result

def plot(xy, param):
    # plt.rcParams['figure.figsize'] = (14.0, 14.0)
    colors = ['blue', 'green', 'red', 'cyan', 'yellow', 'magenta']

    plt.subplot(2, 1, 1)
    # print function
    xy = np.array(xy)
    x = np.linspace(np.min(xy, axis=0)[0]-1, np.max(xy, axis=0)[0]+1, 3001)
    plt.plot(x, f(x, param[0]), color='black')

    # print points
    plt.scatter(xy[:, 0], xy[:, 1], color='red')

    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,x2,-100,100))

    plt.subplot(2, 1, 2)
    # print function
    x = np.linspace(np.min(xy, axis=0)[0]-1, np.max(xy, axis=0)[0]+1, 3001)
    plt.plot(x, f(x, param[1]), color='black')

    # print points
    plt.scatter(xy[:, 0], xy[:, 1], color='red')


    # x1,x2,y1,y2 = plt.axis()
    # plt.axis((x1,x2,-100,100))

    plt.show()

def get_data(filename):
    xy = []
    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')
        for item in data:
            xy.append(list(map(float, item.split(','))))

    return xy

if __name__ == '__main__':
    xy = get_data('./testfile.txt')
    plot(xy, [4.43295031, 29.3064])
