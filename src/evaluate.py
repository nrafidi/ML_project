from os import path
import re

import numpy as np
from scipy.cluster import vq

import flow
import util
import video

## compute descriptors for interest points in a video file, saving to
## disk and not recomputing if already saved
def get_file_descs(fn):
    f = path.join('mats', path.basename(fn) + '.npy')
    if path.exists(f):
        descs = np.load(f)
    else:
        frames = util.vidread(fn)
        descs = video.compute_descriptors(frames)
        np.save(f, descs)

    return descs

labels = []
all_descs = []
n_descs = [0]
for fn in open('train_files.txt'):
    fn = fn.strip()
    print fn

    descs = get_file_descs(fn)

    all_descs.append(descs)
    n_descs.append(descs.shape[0])

    label = re.match('[^/]*/([^/]*)', fn).group(1)
    labels.append(label)

label_set = sorted(set(labels))
labels = np.searchsorted(label_set, labels)

all_descs = np.vstack(all_descs)

np.save('descs.npy', all_descs)
print 'all_descs:', all_descs.shape

## cluster to produce dictionary of spatiotemporal words
print 'clustering...'
if path.exists('clusters.npy'):
    clusters = np.load('clusters.npy')
    cluster_inds = np.load('cluster_inds.npy')
    dict_size = clusters.shape[0]
else:
    dict_size = 300
    clusters, cluster_inds = vq.kmeans2(all_descs, dict_size, iter=20, minit='points')

    np.save('clusters.npy', clusters)
    np.save('cluster_inds.npy', cluster_inds)

## produce histograms for each training video
count_edges = np.cumsum(n_descs)
print count_edges, labels
video_hists = []
for a, b in zip(count_edges[:-1], count_edges[1:]):
    hist = np.histogram(cluster_inds[a:b], np.arange(dict_size + 1))[0]
    hist = hist.astype(np.float64) / hist.sum()
    #print hist
    video_hists.append(hist)

video_hists = np.vstack(video_hists)

m = n = 0
for fn in open('test_files.txt'):
    fn = fn.strip()
    descs = get_file_descs(fn)

    words = vq.vq(descs, clusters)[0]

    hist = np.histogram(words, np.arange(dict_size + 1))[0]
    hist = hist.astype(np.float64) / hist.sum()
    #print hist

    #print hist.shape, video_hists.shape

    neigh_ind = vq.vq(hist[None,:], video_hists)[0][0]
    print fn
    print label_set[labels[neigh_ind]]
    pred_label = re.match('[^/]*/([^/]*)', fn).group(1)

    m += pred_label == label_set[labels[neigh_ind]]
    n += 1

print m, n, m / float(n)
