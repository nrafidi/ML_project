#Given a query point and two other points, finds which point (A or B)
#is closest to the query point, by using the sum of all the distances
#returned by get_dist

import get_distance

def nearest(query, A, B):
    distA = sum(get_dist(query, A))
    distB = sum(get_dist(query, B))
    if (distA > distB):
        return B
    else:
        return A
