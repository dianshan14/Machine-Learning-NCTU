import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from utils import get_MNIST_data, get_parsed_MNIST, get_binned_MNIST

def switch(inp):
    if inp == 'True' or inp == 'true':
        return True
    else:
        return False

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0, choices=[0, 1])
parser.add_argument('--print_post_num', type=int, default=1, choices=[1, 5, 10, 784])
parser.add_argument('--plot_imagination', type=switch, default=False)
parser.add_argument('--smooth', type=switch, default=False)
args = parser.parse_args()
print(args, '\n')

def show_image(img):
    plt.imshow(img, cmap='Greys_r')
    plt.show()

def bin_pixel(pixels, num_of_bins=32):
    OFFSET = 8
    for i in range(num_of_bins):
        # mask = i * OFFSET <= pixels
        # mask *= (pixels < i * OFFSET + OFFSET)
        # pixels[mask] = i
        mask = np.logical_and(i * OFFSET <= pixels, pixels < i * OFFSET + OFFSET)
        pixels[mask] = i

def Gaussian(x, mu, var):
    return (1/np.sqrt(2*np.pi*var))*np.exp(-((x-mu)**2)/(2*var))

def print_posterior(posterior):
    for post_i in range(args.print_post_num):
        print('Posterior (in log scale)')
        for i, v in enumerate(posterior[post_i]):
            print('%s: %s' % (i, v / np.sum(posterior[post_i])))
        print('Prediction: %s, Ans: %s' % (y_pred[post_i], y_test[post_i]))
        # show_image(x_test[post_i].reshape(28, 28))
        print()

def print_imagination(imagination):
    """
    imagination: (10, 28, 28)
    """
    imagination = imagination.astype(np.int)
    print('Imagination of numbers in Bayesian classifier:\n')
    for i, v in enumerate(imagination):
        print('%s:' % i)
        for j in range(v.shape[0]):
            print(v[j])
    print()

print('%s : Naive bayes classifier for %s\n' % (args.mode, 'discrete mode' if args.mode == 0 else 'continuous mode'))
x_train, y_train, x_test, y_test = get_MNIST_data('./data')

if args.mode == 0:
    print('\nIn discrete mode, start binnify 256 to 32')
    bin_pixel(x_train)
    bin_pixel(x_test)
    print('Binnify completely.')

# calculate prior
# -------------------------------------------------------------- #
_, count_of_classes = np.unique(y_train, return_counts=True)
prior = count_of_classes / x_train.shape[0]
# -------------------------------------------------------------- #


# calculate likelihood
# -------------------------------------------------------------- #
if args.mode == 0:
    P_Ai_given_Ck = np.ones((10, 784, 32))

    for i in range(10):
        # take over images of particular class
        Ck = x_train[y_train == i, :]
        for j in range(784):
            # count per pixel stats for classes
            unique, count = np.unique(Ck[:, j], return_counts=True)
            P_Ai_given_Ck[i, j, unique] += count

    for i in range(10):
        P_Ai_given_Ck[i, :, :] /= (count_of_classes[i] + 1)

    imagination = np.argmax(P_Ai_given_Ck, axis=2).reshape(-1, 28, 28)
    imagination[imagination < 16] = 0
    imagination[imagination >= 16] = 1

else:
    P_Ai_given_Ck = np.empty((10, 784, 2))

    for i in range(10):
        # take over images of particular class
        Ck = x_train[y_train == i, :]
        # dimension reduction on pixel-axis (784)
        P_Ai_given_Ck[i, :, 0] = mu = np.mean(Ck, axis=0)
        # P_Ai_given_Ck[i, :, 1] = np.mean(Ck ** 2, axis=0) - mu ** 2 + 1700
        P_Ai_given_Ck[i, :, 1] = np.var(Ck, axis=0) + 1700


    # It will cuase fatal error, if without copy() here.
    imagination = P_Ai_given_Ck[:, :, 0].copy().reshape(-1, 28, 28)
    if args.smooth == False:
        imagination[imagination < 128] = 0
        imagination[imagination >= 128] = 1
# -------------------------------------------------------------- #

# take log on prior and likelihood (for purpose of calculating posterior)
# -------------------------------------------------------------- #
prior = np.log(prior)
if args.mode == 0:
    P_Ai_given_Ck = np.log(P_Ai_given_Ck)
# -------------------------------------------------------------- #


# prediction
# -------------------------------------------------------------- #
likelihood = np.empty((10000, 10, 784))
if args.mode == 0:
    for i in range(10):
        for j in range(784):
            likelihood[:, i, j] = P_Ai_given_Ck[i, j, x_test[:, j]]

    likelihood = np.sum(likelihood, axis=2)
else:
    for i in range(10):
        likelihood[:, i, :] = Gaussian(x_test, P_Ai_given_Ck[i, :, 0], P_Ai_given_Ck[i, :, 1])

    # likelihood[likelihood == 0.0] = 1
    likelihood = np.sum(np.log(likelihood), axis=2)

unnorm_posterior = likelihood + prior
posterior = unnorm_posterior - np.sum(unnorm_posterior, axis=1, keepdims=True)
y_pred = np.argmax(posterior, axis=1)
if args.print_post_num > 0:
    print_posterior(posterior)
print_imagination(imagination)
print('Error rate: %s' % (np.mean(y_pred != y_test)))
print('Elapsed time: %s' % (time.time() - start))
if args.plot_imagination == True:
    for i in range(10):
        show_image(imagination[i, ...])
# -------------------------------------------------------------- #
