import multiprocessing
from multiprocessing.managers import BaseManager
import time
import sys

import numpy as np
from sklearn.cross_validation import LeavePOut

from actions import actions

if __name__ == '__main__':
    in_q = multiprocessing.Queue()
    out_q = multiprocessing.Queue()
    class QueueManager(BaseManager): pass
    QueueManager.register('get_in_queue', lambda: in_q)
    QueueManager.register('get_out_queue', lambda: out_q)
    man = QueueManager(('', 5017), authkey='aoeu')
    man.start()

    from actions import actions

    n = 0
    for inds in LeavePOut(len(actions), 2):
        in_q.put(inds)
        n += 1

    print >>sys.stderr, 'queued %d tasks...' % n

    all_scores = np.zeros((actions.size, actions.size))
    class_scores = np.zeros((actions.size, 2))
    for i in range(n):
        preds, labels, classes = out_q.get()

        for c in classes:
            inds = labels == c

            correct = preds[inds] == labels[inds]

            right = correct.sum()
            count = correct.size
            score = right / float(count)

            #print '%s: %f (%d/%d)' % (actions[c], score, right, count)
            class_scores[c, 0] += right
            class_scores[c, 1] += count

            # assume that there are two classes
            other = classes[classes != c][0]
            all_scores[c, other] = score

        print >>sys.stderr, i

    # for (a, (x, y)) in zip(actions, class_scores):
    #     print '%s & %.3f\\\\' % (a, x / y)

    print ','.join('%.4f' % (x / y) for x, y in class_scores)

    with open('res/%d.txt' % int(time.time()), 'w') as out:
        print >>out, '\n'.join(','.join(map(str, r)) for r in all_scores)
    man.shutdown()
