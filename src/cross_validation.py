import csv
import glob
import multiprocessing
from multiprocessing.managers import BaseManager
import os
from os import path

import numpy as np
from scipy.spatial import distance
from scipy.cluster import vq

import lin_reg
import util
import video

VIDEO_DIR = '../videos'
LABEL_DIR = '../labels'

from actions import actions

# read all video labels
all_labels = {}
for action in actions:
    fn = path.join(LABEL_DIR, '%s.txt' % action)
    with open(fn) as f:
        for line in f:
            spl = line.split(',')
            all_labels[spl[0]] = map(int, spl[1:])

class_labels = {}
for action in actions:
    fn = path.join(LABEL_DIR, 'cls', '%s.txt' % action)
    with open(fn) as f:
        class_labels[action] = map(int, f.readline().split(','))

## compute descriptors for interest points in a video file, saving to
## disk and not recomputing if already saved
def get_file_descs(fn):
    #print 'computing descriptors for %s' % fn
    f = path.join('descs', path.basename(fn) + '.npy')
    if path.exists(f):
        descs = np.load(f)
    else:
        frames = util.vidread(fn)
        pts = video.get_interest_points(frames)
        descs = video.compute_descriptors(frames, pts)
        np.save(f, descs)

    return descs

## load descriptors for one action
def load_action(action):
    action_dir = path.join(VIDEO_DIR, action)
    video_fns = glob.glob(path.join(action_dir, '*'))

    n_descs = []
    descs = []
    labels = []

    for fn in video_fns:
        #print fn
        d = get_file_descs(fn)

        n_descs.append(d.shape[0])
        descs.append(d)
        labels.append(all_labels[path.basename(fn)])

    return n_descs, descs, labels

## load descriptors for all actions
def load_actions(actions):
    action_n_descs = [0]
    video_n_descs = [0]
    all_descs = []
    all_labels = []
    for action in actions:
        n_descs, descs, labels = load_action(action)
        video_n_descs.extend(n_descs)
        action_n_descs.append(len(n_descs))
        all_descs.extend(descs)
        all_labels.extend(labels)

    #print actions
    #print action_n_descs, len(action_n_descs)
    #print video_n_descs, len(video_n_descs)
    return action_n_descs, video_n_descs, np.vstack(all_descs), all_labels

# load_actions(actions)
# exit()

## turn OFH descs from some videos into histograms
def get_desc_hists(clusters, descs, n_descs):
    dict_size = clusters.shape[0]
    words = vq.vq(descs, clusters)[0]
    count_edges = np.cumsum(n_descs)
    hists = []
    for a, b in zip(count_edges[:-1], count_edges[1:]):
        hist = np.histogram(words[a:b], np.arange(dict_size + 1))[0]
        hist = hist.astype(np.float64) / hist.sum()
        hists.append(hist)

    return np.vstack(hists)

np.set_printoptions(precision=2)

def run_test(train_inds, test_inds):
    print 'train actions:', actions[train_inds]
    print 'test actions:', actions[test_inds]

    savedir = path.join('save', '_'.join(actions[test_inds]))
    try:
        os.makedirs(savedir)
    except OSError:
        pass

    #### TRAIN

    # load OFH descriptors from training videos from all-but-two classes
    train_action_n, train_video_n, train_descs, train_labels = load_actions(actions[train_inds])

    # cluster and quantize to produce BoW descriptors
    dict_size = 300
    print 'clustering...'
    print 'train_descs:', train_descs.shape
    if path.exists(path.join(savedir, 'clusters.npy')):
        clusters = np.load(path.join(savedir, 'clusters.npy'))
        cluster_inds = np.load(path.join(savedir, 'cluster_inds.npy'))
    else:
        clusters, cluster_inds = vq.kmeans2(train_descs, dict_size, iter=20, minit='points')
        np.save(path.join(savedir, 'clusters.npy'), clusters)
        np.save(path.join(savedir, 'cluster_inds.npy'), cluster_inds)

    # produce quantized histograms for each training video
    # count_edges = np.cumsum(n_descs)
    # train_hists = []
    # for a, b in zip(count_edges[:-1], count_edges[1:]):
    #     hist = np.histogram(cluster_inds[a:b], np.arange(dict_size + 1))[0]
    #     hist = hist.astype(np.float64) / hist.sum()
    #     train_hists.append(hist)
    print 'quantizing...'
    f = path.join(savedir, 'train_hists.npy')
    if path.exists(f):
        train_hists = np.load(f)
    else:
        train_hists = get_desc_hists(clusters, train_descs, train_video_n)
        np.save(f, train_hists)

    # linearly regress for each attribute based on manually produced labels
    print 'training regressors...'
    cls = lin_reg.train(train_hists, train_labels)

    #### TEST
    # load OFH descriptors from test videos from two classes
    test_action_n, test_video_n, test_descs, test_labels = load_actions(actions[test_inds])

    # produce BoW descriptors
    print 'quantizing test...'
    test_hists = get_desc_hists(clusters, test_descs, test_video_n)
    print test_video_n
    print test_hists.shape

    # apply regressors to get all estimated semantic labels
    print 'predicting...'
    sem_features = cls.predict(test_hists)
    for s in sem_features:
        print s
    print 'features:', sem_features.shape

    # get class semantic features for comparison
    print 'class labels:'
    test_class_labels = []
    for ind in test_inds:
        test_class_labels.append(class_labels[actions[ind]])
    test_class_labels = np.array(test_class_labels)

    # find nearer semantic neighbor among two classes
    preds, dists = vq.vq(sem_features, test_class_labels)
    preds = np.array(test_inds)[preds]

    print preds
    print dists

    # # display distance to each class for each sample
    # dists = distance.cdist(sem_features, test_class_labels)
    # dists = np.hstack([dists, (dists[:,0] - dists[:,1])[:,None]])
    # print dists

    labels = np.hstack([np.array([i]*n) for i, n in zip(test_inds, test_action_n[1:])])
    print labels

    print labels == preds

    np.save(path.join(savedir, 'preds.npy'), preds)
    np.save(path.join(savedir, 'labels.npy'), labels)
    np.save(path.join(savedir, 'classes.npy'), np.array(test_inds))

q = multiprocessing.Queue()
class QueueManager(BaseManager): pass
QueueManager.register('get_queue', lambda: q)
man = QueueManager(('', 5017), authkey='aoeu')
man.connect()

q = man.get_queue()
while True:
    print 'getting inds...'
    inds = q.get(10)
    run_test(*inds)
