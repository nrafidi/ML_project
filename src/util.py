import subprocess
import tempfile

import numpy as np

import Image

def vidwrite(frames, fn):
    d = tempfile.mkdtemp()

    for frame_num in xrange(frames.shape[-1]):
        frame = frames[...,frame_num]
        #cv2.imwrite('%03d.jpg' % frame_num, frame)
        Image.fromarray(frame).save('%s/%03d.jpg' % (d, frame_num), quality=95)

    subprocess.call(['ffmpeg', '-y', '-i', '%s/%%03d.jpg' % d, fn])

def rescale(arr):
    arr = (arr - arr.min()) * 255 / (arr.max() - arr.min())
    arr = 128 + .7 * (arr - 128)
    arr = arr.astype(np.uint8)
    return arr

