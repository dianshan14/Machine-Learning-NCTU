# TODO how many iterations to plot visualization
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def set_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perplexity', type=float, default=10)
    parser.add_argument('--max_iter', type=int, default=1000)
    # parser.add_argument('--sne', type=str, default='ssne', choices=['tsne', 'ssne'])
    parser.add_argument('--sne', type=str, default='tsne', choices=['tsne', 'ssne'])
    args = parser.parse_args()
    return args


def visualize_embedding(Y, labels, it, IMG_PATH):
    # if not it:
    #     clean_result()
    title = {'tsne': 't-SNE', 'ssne': 'Symmetric SNE'}[args.sne]
    plt.title('\n'.join([title, 'Perplexity: %d, iteration: %d' % (args.perplexity, it)]))
    plt.scatter(Y[:, 0], Y[:, 1], s=15, c=labels)
    plt.colorbar()
    plt.savefig(os.path.join(IMG_PATH, 'it%04d.png' % (it, )))
    plt.clf()


def make_gif(IMG_PATH):
    imgs_filename = os.listdir(IMG_PATH)
    if 'output.gif' in imgs_filename:
        imgs_filename.remove('output.gif')
    imgs_filename = sorted(imgs_filename, key=lambda x: int(x[x.index('t')+1:x.index('.')]))

    imgs = []
    for filename in imgs_filename:
        imgs.append(Image.open(os.path.join(IMG_PATH, filename)))

    imgs[0].save('%s/output.gif' % IMG_PATH, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=200, loop=0)


def clean_result():
    dir_path = os.path.dirname(__file__) + '/result/%s/perplexity%d' % (SNE_WAY, PERPLEXITY)

    os.makedirs(dir_path, exist_ok=True)

    if os.path.exists(dir_path):
        files = glob(dir_path + '/*.png')

        for f in files:
            os.remove(f)


def visualize_similarity(P, Q, labels):
    P = np.log(P)
    max_value = np.max(P)
    min_value = np.min(P)
    print(max_value, min_value)

    index = np.argsort(labels)
    P = P[index][:, index]
    img = plt.imshow(P, cmap='gray', vmin=min_value, vmax=max_value)
    plt.colorbar(img)
    plt.show()

    Q = np.log(Q)
    max_value = np.max(Q)
    min_value = np.min(Q)
    print(max_value, min_value)

    Q = Q[index][:, index]
    img = plt.imshow(Q, cmap='gray', vmin=min_value, vmax=max_value)
    plt.colorbar(img)
    plt.show()


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, max_iter=1000, IMG_PATH=''):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    store_P_path = os.path.join('storage', 'P_%d_%d_%s.npy' % (perplexity, max_iter, args.sne))
    store_Q_path = os.path.join('storage', 'Q_%d_%d_%s.npy' % (perplexity, max_iter, args.sne))
    if os.path.isfile(store_P_path):
        print('Load pre-computed P')
        P = np.load(store_P_path)
    else:
        print('Pre-computed P does not exist, compute and save it')
        P = x2p(X, 1e-5, perplexity)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.									# early exaggeration
        P = np.maximum(P, 1e-12)

        np.save(store_P_path, P)

    # Run iterations
    for it in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)

        # TODO
        # num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        if args.sne == 'tsne':
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        elif args.sne == 'ssne':
            num = np.exp(-1. * np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            raise NotImplementedError(args.sne)

        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            # TODO
            # dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            if args.sne == 'tsne':
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            elif args.sne == 'ssne':
                dY[i, :] = np.sum(np.tile(PQ[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
            else:
                raise NotImplementedError(args.sne)

        # Perform the update
        if it < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (it + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            # TODO +1
            print("Iteration %d: error is %f" % (it + 1, C))
            # TODO draw_embedding
            visualize_embedding(Y, labels, it + 1, IMG_PATH)

        # Stop lying about P-values
        if it == 100:
            P = P / 4.

    # TODO
    np.save(store_Q_path, Q)
    make_gif(IMG_PATH)
    visualize_similarity(P, Q, labels)


def get_data(path):
    data = np.loadtxt(os.path.join(path, 'mnist2500_X.txt'))
    labels = np.loadtxt(os.path.join(path, 'mnist2500_labels.txt'))
    return data, labels


if __name__ == "__main__":
    # TODO INFO
    """
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")
    Y = tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
    """

    args = set_argument()
    IMG_PATH = 'storage/%s/%s' % (args.sne, args.perplexity)
    if not os.path.isdir(IMG_PATH):
        print('CREATE DIRECTORY')
        os.system('mkdir -p %s' % (IMG_PATH, ))

    args = set_argument()
    data, labels = get_data('.')
    # tsne(data, 2, 50, perplexity=args.perplexity, max_iter=args.max_iter, IMG_PATH=IMG_PATH)
    # make_gif(IMG_PATH)
    P = np.load('storage/P_30_1000_ssne.npy')
    Q = np.load('storage/Q_30_1000_ssne.npy')
    visualize_similarity(P, Q, labels)
