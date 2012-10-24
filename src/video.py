import sys

import cv2
import numpy as np
from scipy.ndimage import filters

import flow
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

    return np.array(max_locs)[:,inds[:n_pts]]
    #return tuple(q[inds[:n_pts]] for q in max_locs)

## TODO the flows seem to be more on the range of \pm 20, so let's
## maybe try that later
bins = np.linspace(-4, 4, 32)
def compute_descriptors(frames, nbhd=(8,8,5)):
    pts = get_interest_points(frames)
    flo = flow.get_flow(frames)

    nx, ny, nt = nbhd
    sx, sy, st = frames.shape

    inds = ((nx <= pts[0,:] < sx - nx) &
            (ny <= pts[1,:] < sy - ny) &
            (nt <= pts[2,:] < st - nt))

    pts = pts[:,inds]

    descs = []
    for x, y, t in zip(*pts):
        sub_flo = flo[x-nx:x+nx+1,
                      y-ny:y+ny+1,
                      t-nt:t+nt+1]

        xhist = np.histogram(sub_flo[...,0], bins=bins)
        yhist = np.histogram(sub_flo[...,1], bins=bins)
        xhist /= xhist.sum()
        yhist /= yhist.sum()

        descs.append(np.hstack([xhist, yhist]))

    return np.vstack(descs)

if __name__ == '__main__':
    frames = util.vidread(sys.argv[1])
    pts = get_interest_points(frames)

    for x, y, t in zip(*pts):
        frames[max(0, x-3):min(frames.shape[0], x+4),
               max(0, y-3):min(frames.shape[1], y+4),
               max(0, t-2):min(frames.shape[2], t+3)] = 255

    util.vidwrite(frames, sys.argv[2])
