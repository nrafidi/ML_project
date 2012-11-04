import multiprocessing
from multiprocessing.managers import BaseManager
import time

from sklearn.cross_validation import LeavePOut


if __name__ == '__main__':
    q = multiprocessing.Queue()
    class QueueManager(BaseManager): pass
    QueueManager.register('get_queue', lambda: q)
    man = QueueManager(('', 5017), authkey='aoeu')
    man.start()

    from actions import actions

    res = []
    for inds in LeavePOut(len(actions), 2):
        q.put(inds)
        #res.append(pool.apply_async(run_test, args=inds))

    # for r in res:
    #     r.get()

    while q.qsize() > 0:
        print q.qsize()
        time.sleep(5)
