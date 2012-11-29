import glob
import numpy as np

n = 0
s = None
for fn in glob.glob('res/*'):
    n += 1
    #a = np.genfromtxt(fn, delimiter=',')
    a = [np.array(map(float, l.split(','))) for l in open(fn)]

    if s is None:
        s = a
    else:
        for x, y in zip(s, a):
            x += y
        #s += a

print '\n'.join(','.join('%.5f' % (c/n) for c in r) for r in s)
