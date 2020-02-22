import struct
import os
import numpy as np
def get_data(filename):
    with open(filename, 'r') as f:
        data = f.read().strip()
    return data.split('\n')

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
