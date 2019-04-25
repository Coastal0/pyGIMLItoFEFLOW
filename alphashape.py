# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:15:59 2018

@author: 264401k
"""

from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

import workbook as w
def is_inside(x, y, points, edges, eps=1.0e-10):
    intersection_counter = 0
    for i, j in edges:
        assert abs((points[i,1]-y)*(points[j,1]-y)) > eps, 'Need to handle these end cases separately'
        y_in_edge_domain = ((points[i,1]-y)*(points[j,1]-y) < 0)
        if y_in_edge_domain:
            upper_ind, lower_ind = (i,j) if (points[i,1]-y) > 0 else (j,i)
            upper_x = points[upper_ind, 0] 
            upper_y = points[upper_ind, 1]
            lower_x = points[lower_ind, 0] 
            lower_y = points[lower_ind, 1]

            # is_left_turn predicate is evaluated with: sign(cross_product(upper-lower, p-lower))
            cross_prod = (upper_x - lower_x)*(y-lower_y) - (upper_y - lower_y)*(x-lower_x)
            assert abs(cross_prod) > eps, 'Need to handle these end cases separately'
            point_is_left_of_segment = (cross_prod > 0.0)
            if point_is_left_of_segment:
                intersection_counter = intersection_counter + 1
    return (intersection_counter % 2) != 0


nodes = w.loadNodes(r"K:\nodesXY_FEPEST.mat")
edges = alpha_shape(nodes, alpha=0.1, only_outer=True)
plt.scatter(nodes[:,0], nodes[:,1], s = 1)
for m, j in edges:
    plt.plot(nodes[[m, j], 0], nodes[[m, j], 1])

#### Rand number test
nodes = np.random.randn(50,2)
n_rows = 1
n_cols = 5
fig, ax = plt.subplots(n_rows,n_cols,squeeze=True, sharex = True, sharey = True)
plt.axis('equal')
for n,i in enumerate(np.linspace(0.5,1.5,n_cols*n_rows)):
    edges = alpha_shape(nodes, alpha=i, only_outer=True)
    ax[n].scatter(nodes[:,0], nodes[:,1])
    for m, j in edges:
        ax[n].plot(nodes[[m, j], 0], nodes[[m, j], 1])
