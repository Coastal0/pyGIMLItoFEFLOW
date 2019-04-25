# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:04:39 2017

@author: 264401k
"""

from scipy.spatial import Delaunay
import numpy as np
import matplotlib as mpl

import math, warnings
from shapely.geometry import MultiLineString, Polygon, Point, MultiPoint, MultiPolygon
from shapely.ops import cascaded_union, polygonize

def _get_radii(points, tri_vertices):
    """Compute circum-circle radii of triangles
    
    points : array-like (n, 2) : collection of points
    tri_vertices : array-like of shape (m, 3) : Delaunay triangle vertices (indices for points)
    returns: array-like (m)
    
    inspired from: https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
    """
    radii = np.empty(len(tri_vertices))
    radii.fill(np.inf)

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for i, v in enumerate(tri_vertices):
        ia, ib, ic = v
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        if  s*(s-a)*(s-b)*(s-c) < 0 : continue
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        if area == 0: continue

        circum_r = a*b*c/(4.0*area)
        radii[i] = circum_r
        
    return radii
    
    
def is_outlier(x, thresh=3.5):
    """
    Returns a boolean array with True if x are outliers and False 
    otherwise.

    Parameters:
    -----------
        x : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 

        Found there:
        http://stackoverflow.com/a/22357811/2192272
    """
    if len(x.shape) == 1:
        x = x[:,None]
    median = np.median(x, axis=0)
    diff = np.sum((x - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def _make_polygon(points, tri_vertices):
    """Make polygon out of edges (e.g. from a Delaunay triangulation)
    
    Return: Shapely Polygon or MultiPolygon
    
    inspired from: https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
    """
    edge_points = []

    def _add_edge(i, j):
        """Add a line between the i-th and j-th points"""
        edge_points.append(points[ [i, j] ])

    # loop over triangles: 
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri_vertices:
        _add_edge(ia, ib)
        _add_edge(ib, ic)
        _add_edge(ic, ia)

    m = MultiLineString(edge_points)
    subpoly = list(polygonize(m))
    poly = cascaded_union(subpoly)
    return poly


def get_concave_hull(points, thres=3.5):
    """
    Parameters
    ----------
    points: shapely line or (n, 2) array of points
    thresh : passed to is_outlier 
    
    Returns
    -------
    shapely's Polygon or MultiPolygon (if disjoint polygons)
        if a MultiPolygon is returned, this is a problem...
        Simplify geometry in the first place, or use a higher threshold.
    """
    points = np.asarray(points)
    tri = Delaunay(points)
    radii = _get_radii(points, tri.vertices)
    outliers = is_outlier(radii, thres)
    poly = _make_polygon(points, tri.vertices[~outliers])
    if isinstance(poly, MultiPolygon):
        warnings.warn('disjoint polygons: use higher threshold (hint: simplify geometry)')
    return poly.boundary

# simply geometry without tolerance
# speed up things (otherwise final polygons might be disjoint)
    
data=np.loadtxt('test.dat', skiprows = 1)
outl = data[:, [1,2]]
hull = get_concave_hull(outl)
import shapely
shapely.geometry.collection.GeometryCollection([Polygon(outl), hull])

hull = get_concave_hull(outl, thres=1)
shapely.geometry.collection.GeometryCollection([Polygon(outl), hull])
