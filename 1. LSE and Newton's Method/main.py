import argparse
import copy
from utils import plot, get_data

class Matrix(object):
    def __init__(self, *args, **kwargs):
        self.row = args[0]
        self.col = args[1]
        self.matrix = [[0 for i in range(self.col)] for j in range(self.row)]

    @classmethod
    def Identity(cls, row, strength=1):
        identity = cls(row, row)
        for i in range(row):
            identity.matrix[i][i] = strength

        return identity

    @classmethod
    def init_by_xy(cls, xy, n):
        A = cls(len(xy), n)
        b = cls(len(xy), 1)
        for i in range(A.row):
            b.matrix[i][0] = xy[i][1]
            for j in range(A.col):
                A.matrix[i][j] = xy[i][0] ** (n - j - 1)

        return A, b

    @classmethod
    def init_by_list(cls, array):
        row, col = len(array), len(array[0])
        result = cls(row, col)
        for i in range(row):
            for j in range(col):
                result.matrix[i][j] = array[i][j]
        return result

    @classmethod
    def Elementary(cls, row, i, j, factor):
        E = Matrix.Identity(row)
        E.matrix[j][i] = -factor
        E_inv = Matrix.Identity(row)
        E_inv.matrix[j][i] = factor
        return E, E_inv

    def __add__(self, matrix):
        assert self.row == matrix.row and self.col == matrix.col, 'Size error.'
        for i in range(self.row):
            for j in range(self.col):
                self.matrix[i][j] += matrix.matrix[i][j]
        return self

    def __sub__(self, matrix):
        assert self.row == matrix.row and self.col == matrix.col, 'Size error.'
        for i in range(self.row):
            for j in range(self.col):
                self.matrix[i][j] -= matrix.matrix[i][j]
        return self

    def __mul__(self, matrix):
        assert self.col == matrix.row, 'Size error.'
        result = Matrix(self.row, matrix.col)
        for i in range(result.row):
            for j in range(result.col):
                tmp = 0
                for k in range(self.col):
                    tmp += self.matrix[i][k] * matrix.matrix[k][j]
                result.matrix[i][j] = tmp

        return result

    def scale(self, factor):
        for i in range(self.row):
            for j in range(self.col):
                self.matrix[i][j] *= factor
        return self

    def _solve(self, L, U, b, n):
        b = self._solve_equation(L.matrix, b, n, (n, ))
        return self._solve_equation(U.matrix, b, n, (n-1, -1, -1))

    def _solve_equation(self, A, b, n, enum):
        assert type(enum) == tuple
        y = [1] * n
        for i in range(*enum):
            y[i] = b[i]
            for j in range(*enum):
                if i != j:
                    y[i] -= A[i][j] * y[j]
            y[i] /= A[i][i]
        return y

    @property
    def T(self):
        result = Matrix(self.col, self.row)
        for i in range(self.row):
            for j in range(self.col):
                result.matrix[j][i] = self.matrix[i][j]
        return result

    @property
    def size(self):
        return (self.row, self.col)

    @property
    def LU_decomposition(self):
        row = self.row
        L = Matrix.Identity(row)
        U = copy.deepcopy(self) # initilize A
        for i in range(row-1):
            for j in range(i+1, row):
                E, E_inv = Matrix.Elementary(row, i, j, U.matrix[j][i]/U.matrix[i][i])
                U = E * U
                L *= E_inv
        # print(U)
        return L, U

    @property
    def inverse(self):
        L, U = self.LU_decomposition
        n = L.row
        result = []
        for i in range(n):
            I_vec = [0] * n
            I_vec[i] = 1
            result.append(self._solve(L, U, I_vec, n))
        return Matrix.init_by_list(result)

    @property
    def to_list(self):
        return self.matrix

    @property
    def item(self):
        assert self.row == 1 and self.col == 1, 'Matrix have more than one element!'
        return self.matrix[0][0]

    def __str__(self):
        for i in range(self.row):
            print(self.matrix[i])
        return ''

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='testfile.txt')
parser.add_argument('--n', type=int, default=3)
parser.add_argument('--strength', type=float, default=0.0)
arg = parser.parse_args()
print(arg, '\n')

# Ax - b = 0
# A: design matrix
# [x0^2, x0^1, 1
#  x1^2, x1^1, 1
#    .    .    .
#    .    .    .
#    .    .    .
#  xn^2, xn^1, 1]

# x: parameter vector
# [a
#  b
#  c]

# b: target vector
# [y0
#  y1
#  y2]

# LSE
# x = (A^T*A + lambda*I)^(-1)*A^T*b

xy = get_data(arg.filename)
A, b = Matrix.init_by_xy(xy, arg.n)
lambda_identity = Matrix.Identity(A.col, arg.strength)

ATA_I = A.T * A + lambda_identity

# L, U = ATA_I.LU_decomposition
# print(L, '\n', U)

# print(ATA_I * ATA_I.inverse)
parameter = ATA_I.inverse * A.T * b
paramter_list = parameter.to_list

print('\x1b[0;30;44mLSE\x1b[0m' + '\nFitting line:')
for i in range(arg.n):
    print('%s X^%s %s' % (paramter_list[i][0], arg.n - i - 1, '' if i == arg.n-1 else '+'))

print('Total error: %s\n' % (((A*parameter-b).T * (A*parameter-b)).item + arg.strength * (parameter.T * parameter).item))

# ------------------------------------- Newton's Method ----------------------------------------- #
print("\x1b[0;30;44mNewton's Method's method\x1b[0m" + "\nFitting line:")
x = Matrix.init_by_list([[1]] * arg.n)
for i in range(5):
    # print(x)
    gradient = (A.T * A * x).scale(2) - (A.T * b).scale(2)
    hessian_inv = (A.T * A).inverse.scale(1/2)
    x -= hessian_inv * gradient

x_list = x.to_list
for i in range(arg.n):
    print('%s X^%s %s' % (x_list[i][0], arg.n - i - 1, '' if i == arg.n-1 else '+'))

print('Total error: %s\n' % (((A*x-b).T * (A*x-b)).item))
plot(xy, [paramter_list, x_list])
