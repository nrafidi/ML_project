import os
from os import path
import sys

import numpy as np

from actions import actions

class_scores = np.zeros((actions.size, 2))

for t in sorted(os.listdir('save')):
    d = path.join('save', t)
    classes = np.load(path.join(d, 'classes.npy'))
    preds = np.load(path.join(d, 'preds.npy'))
    labels = np.load(path.join(d, 'labels.npy'))

    #print t
    #print preds == labels

    #if preds.size < 120: continue

    for c in classes:
        inds = labels == c

        correct = preds[inds] == labels[inds]

        right = correct.sum()
        count = correct.size

        print '%s: %f (%d/%d)' % (actions[c], right / float(count), right, count)
        class_scores[c, 0] += right
        class_scores[c, 1] += count

    print

for (a, (x, y)) in zip(actions, class_scores):
    print >>sys.stderr, '%s & %.3f\\\\' % (a, x / y)
