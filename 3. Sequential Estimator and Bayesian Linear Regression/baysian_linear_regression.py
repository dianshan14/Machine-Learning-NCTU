import argparse
import numpy as np
from utils import poly_basis_generator, plot, extract_standard_deviation

parser = argparse.ArgumentParser()
parser.add_argument('--b', type=float, default=1.0, help='b')
parser.add_argument('--n', type=int, default=4, help='n')
parser.add_argument('--a', type=float, default=1.0, help='a')
parser.add_argument('--w', type=str, default='1,2,3,4', help='w')
args = parser.parse_args()

w = eval('[' + args.w + ']')
print(w)

# prior mean
prior_m = np.zeros((args.n, 1)) # n, 1

# prior posterior
prior_S = np.identity(args.n) * args.b # inverse prior
print(prior_S)

xy = []
ms = [[np.array(w), np.identity(args.n) * args.a], []]

cnt = 0

while True:
    cnt += 1
    x, y = poly_basis_generator(args.n, w, args.a)
    xy.append([x, y])
    X = np.array([[x ** j for j in range(args.n)]])

    inv_posterior_S = prior_S + (1 / args.a) * X.T.dot(X) # posterior_S and prior_S are always inverse of S
    posterior_S = np.linalg.inv(inv_posterior_S)

    posterior_m = posterior_S.dot(prior_S.dot(prior_m) + (1 / args.a) * X.T * y)

    predictive_m = X.dot(posterior_m)
    predictive_S = args.a + X.dot(posterior_S.dot(X.T))

    print('Add data point (%s, %s):\n' % (x, y))
    print('Posterior mean:\n', posterior_m, '\n')
    print('Posterior variance:\n', posterior_S, '\n')
    print('Predictive distribution ~ N(%s, %s)' % (predictive_m.item(), predictive_S.item()))
    print('-'*70)

    # update
    prior_m = posterior_m
    prior_S = inv_posterior_S
    if cnt == 10 or cnt == 50:
        ms.append([posterior_m.reshape(-1), posterior_S])
    if cnt > 1000 and abs(predictive_S.item() - args.a) < 1e-3:
        ms[1] = [posterior_m.reshape(-1), posterior_S]
        print('Update times: ', cnt)
        break

plot(xy, ms, args.a)
