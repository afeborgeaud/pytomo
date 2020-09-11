import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def complete_vertices(vertices):
    ndim = vertices.shape[1]
    nver = 2**ndim
    if vertices.shape[0] == nver:
        return vertices
    else:
        vertices_complete = np.array((nver, ndim), dtype=float)
        for i in vertices.shape[0]:
            vertices_complete[i] = vertices[i]
        

def find_bounds(vor, ivers):
    ivers_ = [i for i in ivers if i != -1]
    if not ivers_:
        return None
    vers = vor.vertices[ivers_]
    mins = vers.min(axis=0)
    maxs = vers.max(axis=0)
    bounds = np.vstack((mins, maxs)).T

    min_bound = vor.points.min(axis=0)
    max_bound = vor.points.max(axis=0)

    for i, bound in bounds:
        if bound[]
    
    return bounds

points = np.array(
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
    [2, 0], [2, 1], [2, 2]])

vor = Voronoi(points)
ndim = vor.points.shape[1]

point_bounds = np.zeros((len(points), ndim, 2), dtype=float)
for ip, ireg in enumerate(vor.point_region):
    ivers = vor.regions[ireg]
    bounds = find_bounds(vor, ivers)
    point_bounds[ip] = bounds

print(point_bounds)

