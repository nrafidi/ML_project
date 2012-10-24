import itertools
import subprocess
import sys
import tempfile

import cv2
import Image
import numpy as np
from scipy.ndimage import filters

def get_flow(frames):
    flows = []
    for frame_num in xrange(frames.shape[-1]-1):
        f1 = frames[...,frame_num]
        f2 = frames[...,frame_num+1]

        flow = cv2.calcOpticalFlowFarneback(f1, f2, flow=None,
                                            pyr_scale=.5, levels=5, winsize=5, iterations=4,
                                            poly_n=5, poly_sigma=1.1, flags=0)

        flows.append(flow[:,:,None,:])

    ## duplicate the last one so we have the same number of frames as
    ## the input video
    flows.append(flows[-1])
    flows = np.concatenate(flows, axis=2)

    return flows

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
        ## compute a dense flow, since sparse flow method doesn't seem
        ## to work
        # TODO figure out what these parameters should be; these seem
        # to provide okay results
        diff = cv2.calcOpticalFlowFarneback(f1, f2, flow=None,
                                            pyr_scale=.5, levels=5, winsize=5, iterations=4,
                                            poly_n=5, poly_sigma=1.1, flags=0)
        #out_frame = np.sqrt((diff**2).sum(2)) * 15
        print i, diff[...,0].min(), diff[...,0].max()
        out_frame = diff[...,0] * 10 + 128

        out_frame[out_frame < 0] = 0
        out_frame[out_frame > 255] = 255
        out_frame = out_frame.reshape(f1.shape).astype(np.uint8)
        Image.fromarray(out_frame).save('%s/%03d.jpg' % (d, i), quality=95)

        f1 = f2
        f2 = getframe(vid)
        if f2 is None: break

    subprocess.call(['ffmpeg', '-y', '-i', '%s/%%03d.jpg' % d, sys.argv[2]])
