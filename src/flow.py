import itertools
import subprocess
import sys
import tempfile

import cv2
import Image
import numpy as np
from scipy.ndimage import filters

def getframe(vid):
    _, f = vid.read()
    if f is None: return f
    return f.mean(2).astype(np.uint8)

if __name__ == '__main__':
    vid = cv2.VideoCapture(sys.argv[1])
    frames = []
    d = tempfile.mkdtemp()

    f1 = getframe(vid)
    f2 = getframe(vid)
    for i in itertools.count():
        ## to compute sparse flow -- I want this to work but it
        ## actually gives nonsensical results
        # xx, yy = np.indices((120,160))
        # xx = xx.ravel()[:,None]
        # yy = yy.ravel()[:,None]
        # pts = np.hstack([xx, yy]).astype(np.float32)
        # next_pts, status, err = cv2.calcOpticalFlowPyrLK(f1, f2, pts, None)
        # diff = next_pts - pts
        # dist = np.sqrt((diff**2).sum(1))
        # dist = f2[xx, yy]

        ## compute a dense flow instead
        # TODO figure out what these parameters should be; these seem
        # to provide okay results
        diff = cv2.calcOpticalFlowFarneback(f1, f2, flow=None,
                                            pyr_scale=.5, levels=5, winsize=5, iterations=4,
                                            poly_n=5, poly_sigma=1.1, flags=0)
        dist = np.sqrt((diff**2).sum(2)) * 15

        print diff.shape, dist.shape
        print i, dist.max()

        dist[dist > 255] = 255
        dist = dist.reshape(f1.shape).astype(np.uint8)
        Image.fromarray(dist).save('%s/%03d.jpg' % (d, i), quality=95)

        f1 = f2
        f2 = getframe(vid)
        if f2 is None: break

    subprocess.call(['ffmpeg', '-y', '-i', '%s/%%03d.jpg' % d, sys.argv[2]])
