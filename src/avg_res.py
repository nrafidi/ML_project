import glob
import numpy as np

n = 0
s = None
for fn in glob.glob('res/*'):
    n += 1
    a = np.genfromtxt(fn, delimiter=',')
    if s is None:
        s = a
    else:
        s += a

print '\n'.join(','.join('%.5f' % c for c in r) for r in s / n)
