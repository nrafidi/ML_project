#Calculates Euclidean, Manhattan and Cosine distance between two points and
#Returns all three distances in a list

from math import sqrt

def get_dist(pt1, pt2):
    ed = euclid(pt1, pt2)
    md = manhat(pt1, pt2)
    cd = cosine(pt1, pt2)
    return (ed, md, cd)

def euclid(ep1, ep2):
    edist = 0
    for i, p in enumerate(ep1):
        edist += (p - ep2[i])*(p - ep2[i])
    return sqrt(edist)

def manhat(mp1, mp2):
    mdist = 0
    for i, p in enumerate(mp1):
        mdist += abs(p-mp2[i])
    return mdist

def cosine(p1, p2):
    dist = 0
    Z = [];
    for i in range(len(p1)):
        Z.append(0)
    denom = euclid(Z, p1)*euclid(Z, p2)
    for i, p in enumerate(p1):
        dist += (p*p2[i])
    return dist/denom
