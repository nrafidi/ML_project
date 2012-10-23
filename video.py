import sys

import cv2
import numpy as np
from scipy.ndimage import filters

import util

def local_maxes(arr, rad):
    # TODO check edge mode (default 'reflect' should be fine)
    maxs = filters.maximum_filter(arr, size=2*rad+1)
    locs = arr >= maxs

    return np.nonzero(locs)

def get_interest_points(arr, sigma=1, tau=1, k=5e-4, n_pts=400):
    arr = arr.astype(np.float64)
    s1 = np.array([sigma, sigma, tau])
    s2 = np.sqrt(2) * s1

    blurred = filters.gaussian_filter(arr, s1)
    lx, ly, lt = np.gradient(blurred)

    xx = filters.gaussian_filter(lx * lx, s2)
    yy = filters.gaussian_filter(ly * ly, s2)
    tt = filters.gaussian_filter(lt * lt, s2)
    xy = filters.gaussian_filter(lx * ly, s2)
    xt = filters.gaussian_filter(lx * lt, s2)
    yt = filters.gaussian_filter(ly * lt, s2)

    det = tt * xx * yy + 2 * xt * xy * yt - yy * xt * xt - xx * yt * yt - tt * xy * xy
    tr = tt + xx + yy

    H = det - k * tr**3

    # util.vidwrite(rescale(H), 'h.avi')

    max_locs = local_maxes(H, 3)
    inds = np.argsort(-H[max_locs])
    return tuple(q[inds[:n_pts]] for q in max_locs)

if __name__ == '__main__':
    vid = cv2.VideoCapture(sys.argv[1])
    frames = []
    while True:
        _, frame = vid.read()
        if frame is None: break

        if frame.ndim == 3:
            frame = frame.mean(2).astype(frame.dtype)
        frames.append(frame)

    frames = np.dstack(frames)
    pts = get_interest_points(frames)

    for x, y, t in zip(*pts):
        frames[max(0, x-3):min(frames.shape[0], x+4),
               max(0, y-3):min(frames.shape[1], y+4),
               max(0, t-2):min(frames.shape[2], t+3)] = 255

    util.vidwrite(frames, sys.argv[2])
