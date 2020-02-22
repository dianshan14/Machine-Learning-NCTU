import argparse
from utils import get_data

def combinations(n, r):
    base = min(n - r , r)
    result = 1.0
    for i in range(1, base+1):
        result *= (n - i + 1) / i
    return result

def bernoulli(head, tail):
    total = head + tail
    return ((head / total) ** head) * ((tail / total) ** tail)

def binomial(head, tail):
    return combinations(head + tail, head) * bernoulli(head, tail)

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='testfile.txt')
parser.add_argument('--a', type=int, default=0)
parser.add_argument('--b', type=int, default=0)
args = parser.parse_args()
print(args)

trials = get_data(args.filename)
a, b = args.a, args.b

for case, trial in enumerate(trials, 1):
    print('case %s: %s' % (case, trial))
    head = trial.count('1')
    tail = len(trial) - head
    print('Likelihood: %s' % binomial(head, tail))
    print('Beta prior:     a = %s b = %s' % (a, b))
    a += head
    b += tail
    print('Beta posterior: a = %s b = %s\n' % (a, b))
