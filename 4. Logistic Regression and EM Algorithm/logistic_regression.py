import numpy as np
import argparse
import random

from utils import uni_gaussian_generator, plot

PREV_NEWTON_NORM = 0
def generate_data_point():
    class_theta = [[[mx1, vx1], [my1, vy1]], [[mx2, vx2], [my2, vy2]]]
    A, y = [], []
    GT = [[], []]
    for i in range(2): # 2 classes
        for j in range(N):
            point = [uni_gaussian_generator(*class_theta[i][0]), uni_gaussian_generator(*class_theta[i][1])]
            GT[i].append(point)
            A.append(point + [1])
            y.append(float(i))
    GT[0] = np.array(GT[0])
    GT[1] = np.array(GT[1])
    # random.shuffle(A)
    # print(A)
    A = np.array(A)
    y = np.array(y)
    return np.array(GT), A, y

def gradient(w, A, y):
    """
    w(phi, 1)
    A(n, phi)
    y(n, 1)
    """
    # e_term = np.array([1 / (1 + np.exp(-A[i].dot(w))) for i in range(2 * N)])
    e_term = 1 / (1 + np.exp(-A.dot(w)))
    # print('e_term', e_term.shape)
    in_term = y[:, None] - e_term
    # print('in_term', in_term.shape)
    result = A.T.dot(in_term)
    # print('gradient', result.shape)
    return result

def inv_Hessian(w, A):
    # D (2N, 2N)

    D = np.zeros((2 * N, 2 * N))
    for i in range(2 * N):
        e_term = np.exp(-A[i].dot(w))
        # print('e_term', e_term)
        D[i][i] = e_term / np.square(1 + e_term)

    Hessian = A.T.dot(D).dot(A)
    if np.linalg.det(Hessian) <= 1e-7:
        return False, None

    # check invertible
    # sum_diagonal = np.sum(Hessian[range(nW), range(nW)]) / (2 * N)
    # if sum_diagonal <= 1e-2:
    #     print('Can not be inverted')
    #     return False
    inv_hessian = np.linalg.inv(Hessian)
    # if float('nan') in inv_hessian[range(nW), range(nW)]:
    #     return False

    return True, inv_hessian

def gradient_descent(w, A, y, lr=0.01):
    """
    w(phi, 1)
    A(n, phi)
    y(n, 1)
    maximize negative log likelihood?
    +
    """
    gradient_vector = gradient(w, A, y)
    norm = np.linalg.norm(gradient_vector)
    # print(norm)
    if norm <= 1e-2:
        return True, w
    else:
        w = w + lr * gradient_vector
        # print('descent result', w.shape)
        return False, w

def Newton_method(w, A, y, lr=0.01):
    global PREV_NEWTON_NORM
    invertible, inv_hessian = inv_Hessian(w, A)
    if invertible:
        # print('Invertible')
        norm = np.linalg.norm(inv_hessian)
        # print(norm)
        if abs(norm - PREV_NEWTON_NORM)  <= 1e-3:
            return True, w
        else:
            w = w + lr * inv_hessian.dot(gradient(w, A, y))
            PREV_NEWTON_NORM = norm
    else:
        # print('uninvertible')
        _, w = gradient_descent(w, A, y, lr=lr)
    # print('newton result', w.shape)
    return False, w

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=50)
parser.add_argument('--case', type=int, default=0, choices=[0, 1])
args = parser.parse_args()


N = [50, 50][args.case]
mx1 = my1 = [1, 1][args.case]
mx2 = my2 = [10, 3][args.case]
vx1 = vy1 = [2, 2][args.case]
vx2 = vy2 = [2, 4][args.case]
# print('Input:', N, mx1, mx2, vx1, vx2)

# hyperparameter
lr = 0.01

# INFO: initialize
w = np.array([0.0, 0.0, 0.0])[:, None]
nW = len(w)

# prepare data format
# A = np.array([[x ** j for j in range(nW)]])
GT, A, y = generate_data_point()
# print('GT', GT.shape)
# print('A', A.shape)
# print('y', y.shape)
# print(A)
# plot(GT, GT, GT)

# learning loop
it = 0
while(True):
    converge, w = gradient_descent(w, A, y, lr=lr)
    if converge:
        break
    it += 1

# predict result
y_pred = 1 / (1 + np.exp(-A.dot(w)))
# print((y_pred[y_pred < 0.5]).shape)
# print((y_pred[y_pred >= 0.5]).shape)
c0 = np.array(A[(y_pred < 0.5).squeeze()][:, :2])
c1 = np.array(A[(y_pred >= 0.5).squeeze()][:, :2])
# print(y_pred)
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
# print(c0.shape)
# print(c1.shape)
# print(y_pred[y_pred < 0.5].shape)
# print(y_pred[y_pred >= 0.5].shape)
GD = np.array([c0, c1])
print()
print('Gradient descent:\n')
print('w:\n', w, '\n')
print('Confusion Matrix:\n\t\tPredict cluster 1  Predict cluster 2')
gt1_cls2 = int(y_pred[y == 0].sum())
gt2_cls2 = int(y_pred[y == 1].sum())
print('In cluster 1\t\t%s\t\t%s' % (N - gt1_cls2, gt1_cls2))
print('In cluster 2\t\t%s\t\t%s\n' % (N - gt2_cls2, gt2_cls2))
print('Sensitivity (Successfully predict cluster 1): %s' % ((N - gt1_cls2) / N))
print('Specificity (Successfully predict cluster 2): %s' % (gt2_cls2 / N))
print('Update times: %s (norm <= 1e-3)' % it)
print('-' * 70)


w = np.array([0.0, 0.0, 0.0])[:, None]
it = 0
while(True):
    converge, w = Newton_method(w, A, y, lr)
    if converge:
        break
    it += 1


y_pred = 1 / (1 + np.exp(-A.dot(w)))
# print((y_pred[y_pred < 0.5]).shape)
# print((y_pred[y_pred >= 0.5]).shape)
c0 = np.array(A[(y_pred < 0.5).squeeze()][:, :2])
c1 = np.array(A[(y_pred >= 0.5).squeeze()][:, :2])
y_pred[y_pred < 0.5] = 0
y_pred[y_pred >= 0.5] = 1
# print(c0.shape)
# print(c1.shape)
NM = np.array([c0, c1])
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

# prepare plotting format
plot(GT, GD, NM)
"""
D1 = red, D2 = blue
GT
	[D1, D2](n, 2N-n) [x, y]
GD

NM
"""
