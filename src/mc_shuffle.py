LABEL_DIR = '../labels'

import numpy as np
from os import path
from actions import actions
from random import shuffle

def get_shuffled():
    # read all video labels
    all_labels = {}
    vid_labels = {}
    mc_vid = {}
    vectors = {}
    i = 0;
    for action in actions:
        fn = path.join(LABEL_DIR, '%s.txt' % action)
        with open(fn) as f:
            for line in f:
                spl = line.split(',')
                vid_labels[i] = spl[0]
                mc_vid[i] = spl[0]
                vectors[i] = map(int, spl[1:])
                all_labels[spl[0]] = map(int, spl[1:])
                i += 1
    #mc_labels = shuffle(all_labels)
    mc_labels = {}
    shuffle(mc_vid)
    shuffle(vectors)
    for j in range(0, i-1, 1):
        mc_labels[mc_vid[i]] = vectors[i] #KeyError: 665

    #For debugging
    for k in range(1, 3):
        print vid_labels[k] + '\n'
        print all_labels[vid_labels[k]] + '\n'
        print mc_labels[vid_labels[k]] + '\n'
    
    
