import glob

import numpy as np
from scipy.stats.mstats import mode

for fn in glob.glob('*.txt'):
    print fn
    vals = []
    for line in open(fn):
        vals.append(map(int, line.split(',')[1:]))
    vals = np.array(vals)
    print vals

    modes, counts = mode(vals)

    print modes
    print (counts / float(vals.shape[0])).min(), (counts / float(vals.shape[0])).argmin()


    with open('cls/%s' % fn, 'w') as f:
        print >>f, ','.join(map(str, modes[0].astype(np.uint8)))
