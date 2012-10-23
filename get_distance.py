import numpy as np

# these also have equivalents in scipy.spatial.distance, but perhaps
# SciPy is less likely to be installed

def get_dist(pt1, pt2):
    ed = euclid(pt1, pt2)
    md = manhat(pt1, pt2)
    cd = cosine(pt1, pt2)
    return (ed, md, cd)

def euclid(ep1, ep2):
    return np.linalg.norm(ep1 - ep2)

def manhat(mp1, mp2):
    return np.sum(np.abs(mp1 - mp2))

def cosine(p1, p2):
    return 1 - np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
