import argparse
import math
from utils import uni_gaussian_generator

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=float, default=3.0, help='mean')
parser.add_argument('--s', type=float, default=5.0, help='variance')
args = parser.parse_args()

def variance(n, add, addsq):
    return (addsq - (add ** 2) / n) / (n - 1)
print('Data point source function: N(%.1f, %.1f)\n' % (args.m, args.s))

n = 1
add = uni_gaussian_generator(args.m, args.s)
addsq = add ** 2
print('Add data point: %f' % (add))
# n = add = addsq = 0

while True:
    point = uni_gaussian_generator(args.m, args.s)
    print('Add data point: %f' % (point))
    n += 1
    add += point
    addsq += point ** 2
    print('Mean = %f   Variance = %f' % (add / n, variance(n, add, addsq)))
    if abs(args.s - variance(n, add, addsq)) <= 1e-4:
        break

print('Update times: %s' % n)
