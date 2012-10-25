import sys

import cv2
import numpy as np
from scipy.ndimage import filters

import flow
import util

## Given an array, return the indices of all locations which are local
## maxima in a square neighborhood. Returns an (arr.ndim)-tuple of
## equal-length vectors, indicating coordinates along each dimension.
def local_maxes(arr, rad):
    # TODO check edge mode (default 'reflect' should be fine)
    maxs = filters.maximum_filter(arr, size=2*rad+1)
    locs = arr >= maxs

    return np.nonzero(locs)

## Given a 3-D array, find Harris interest points. Returns a 3xN array
## of indices, where N is the number of points being returned.
def get_interest_points(arr, sigma=1, tau=1, k=5e-4, n_pts=400):
    arr = arr.astype(np.float64)
    s1 = np.array([sigma, sigma, tau])
    s2 = np.sqrt(2) * s1

    ## find the gradients...
    blurred = filters.gaussian_filter(arr, s1)
    lx, ly, lt = np.gradient(blurred)

    ### ...and integrate their products
    xx = filters.gaussian_filter(lx * lx, s2)
    yy = filters.gaussian_filter(ly * ly, s2)
    tt = filters.gaussian_filter(lt * lt, s2)
    xy = filters.gaussian_filter(lx * ly, s2)
    xt = filters.gaussian_filter(lx * lt, s2)
    yt = filters.gaussian_filter(ly * lt, s2)

    ## determinant/trace of the structure tensor
    det = tt * xx * yy + 2 * xt * xy * yt - yy * xt * xt - xx * yt * yt - tt * xy * xy
    tr = tt + xx + yy

    H = det - k * tr**3

    ## find only local maxima, and sort them by score to return only
    ## the strongest feature points
    max_locs = local_maxes(H, 3)
    inds = np.argsort(-H[max_locs])

    return np.array(max_locs)[:,inds[:n_pts]]

## Given a 3-D array representing a video, find the Harris interest
## points and compute the normalized histograms of optical flow in
## their neighborhoods.

## TODO the flows seem to be more on the range of \pm 20, so let's
## maybe try that later
bins = np.linspace(-4, 4, 33)
bins[0] = -1e10
bins[-1] = 1e10
def compute_descriptors(frames, nbhd=(8,8,5)):
    pts = get_interest_points(frames)
    flo = flow.get_flow(frames)

    nx, ny, nt = nbhd
    sx, sy, st = frames.shape

    ## eliminate points which are too close to the boundaries
    inds = ((nx <= pts[0,:]) & (pts[0,:] < sx - nx) &
            (ny <= pts[1,:]) & (pts[1,:] < sy - ny) &
            (nt <= pts[2,:]) & (pts[2,:] < st - nt))
    pts = pts[:,inds]

    descs = []
    for x, y, t in zip(*pts):
        sub_flo = flo[x-nx:x+nx+1,
                      y-ny:y+ny+1,
                      t-nt:t+nt+1]

        xhist, _ = np.histogram(sub_flo[...,0], bins=bins)
        yhist, _ = np.histogram(sub_flo[...,1], bins=bins)
        xhist = xhist.astype(np.float64) / xhist.sum()
        yhist = xhist.astype(np.float64) / yhist.sum()

        descs.append(np.hstack([xhist, yhist]))

    return np.vstack(descs)

## Given an input video, overlay the Harris points and output to a
## file.
if __name__ == '__main__':
    frames = util.vidread(sys.argv[1])
    pts = get_interest_points(frames)

    for x, y, t in zip(*pts):
        frames[max(0, x-3):min(frames.shape[0], x+4),
               max(0, y-3):min(frames.shape[1], y+4),
               max(0, t-2):min(frames.shape[2], t+3)] = 255

    util.vidwrite(frames, sys.argv[2])
