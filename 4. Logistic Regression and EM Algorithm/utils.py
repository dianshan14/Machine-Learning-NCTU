import numpy as np
import matplotlib.pyplot as plt
import struct
import os

def uni_gaussian_generator(mean, variance):
    normal = np.random.uniform(size=12).sum() - 6
    return normal * np.sqrt(variance) + mean

def plot(ground_truth, gradient_descent, newton_method):
    # plt.rcParams['figure.figsize'] = (14.0, 14.0)
    ground_truth = np.array(ground_truth)
    gradient_descent = np.array(gradient_descent)
    newton_method = np.array(newton_method)
    # x = np.linspace(x_min, x_max, 3001)

    plt.subplot(1, 3, 1)
    # print('Ground Truth: ', ground_truth.shape)
    plt.scatter(ground_truth[0][:, 0], ground_truth[0][:, 1], color='red')
    plt.scatter(ground_truth[1][:, 0], ground_truth[1][:, 1], color='blue')
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, -15, 25))
    plt.title('Ground truth')

    plt.subplot(1, 3, 2)
    # print('Gradient descent', gradient_descent.shape)
    plt.scatter(gradient_descent[0][:, 0], gradient_descent[0][:, 1], color='red')
    plt.scatter(gradient_descent[1][:, 0], gradient_descent[1][:, 1], color='blue')
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, -15, 25))
    plt.title('Gradient Descent')

    plt.subplot(1, 3, 3)
    # print('Newton method', newton_method.shape)
    plt.scatter(newton_method[0][:, 0], newton_method[0][:, 1], color='red')
    plt.scatter(newton_method[1][:, 0], newton_method[1][:, 1], color='blue')
    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, -15, 25))
    plt.title("Newton's method")

    plt.show()


def parse_MNIST(filename, image=True):
    print('Processing %s' % filename)
    with open(filename, 'rb') as f:
        f.seek(0)
        magic = struct.unpack('>4B', f.read(4))
        N = struct.unpack('>I', f.read(4))[0]
        if image:
            H = struct.unpack('>I', f.read(4))[0]
            W = struct.unpack('>I', f.read(4))[0]
            total_bytes = N * H * W * 1
            # data = 255 - np.asarray(struct.unpack('>'+'B'*total_bytes, f.read(total_bytes))).reshape(N, H, W)
            data = np.asarray(struct.unpack('>'+'B'*total_bytes, f.read(total_bytes))).reshape(N, H, W)
        else:
            total_bytes = N * 1
            data = np.asarray(struct.unpack('>'+'B'*total_bytes, f.read(total_bytes))).reshape(N)
        return data


def get_MNIST_data(path):
    x_train = parse_MNIST(os.path.join(path, 'train-images.idx3-ubyte')).reshape(-1, 784)
    y_train = parse_MNIST(os.path.join(path, 'train-labels.idx1-ubyte'), image=False)
    x_test  = parse_MNIST(os.path.join(path, 't10k-images.idx3-ubyte')).reshape(-1, 784)
    y_test  = parse_MNIST(os.path.join(path, 't10k-labels.idx1-ubyte'), image=False)
    print('Parse MNIST dataset completed')
    return x_train, y_train, x_test, y_test

def get_binned_MNIST(path):
    x_train = np.load(os.path.join(path, 'x_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    x_test  = np.load(os.path.join(path, 'x_test.npy'))
    y_test  = np.load(os.path.join(path, 'y_test.npy'))
    print('LOADING NUMPY Finish')
    return x_train, y_train, x_test, y_test

def get_parsed_MNIST(path):
    x_train = np.load(os.path.join(path, 'raw_x_train.npy'))
    y_train = np.load(os.path.join(path, 'raw_y_train.npy'))
    x_test  = np.load(os.path.join(path, 'raw_x_test.npy'))
    y_test  = np.load(os.path.join(path, 'raw_y_test.npy'))
    print('LOADING NUMPY Finish')
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    print(get_data('testfile.txt'))
    data = get_MNIST_data('./data')
    for i in data:
        print(i.shape)

if __name__ == '__main__':
    N = [50, 10]
    mx1 = my1 = 1
    mx2 = my2 = 10
    vx1 = vy1 = 2
    vx2 = vy2 = 2
    class_theta = [[[mx1, vx1], [my1, vy1]], [[mx2, vx2], [my2, vy2]]]
    datapoints = [[], []]
    for i in range(2): # 2 classes
        for j in range(N[i]):
            point = [uni_gaussian_generator(*class_theta[i][0]), uni_gaussian_generator(*class_theta[i][1])]
            datapoints[i].append(point)
    print(len(datapoints[0]))
    print(len(datapoints[1]))
    datapoints[0] = np.array(datapoints[0])
    datapoints[1] = np.array(datapoints[1])
    plot(datapoints, deepcopy(datapoints), deepcopy(datapoints))
