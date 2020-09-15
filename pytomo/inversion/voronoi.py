import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import time

def complete_vertices(vertices):
    ndim = vertices.shape[1]
    nver = 2**ndim
    if vertices.shape[0] == nver:
        return vertices
    else:
        vertices_complete = np.array((nver, ndim), dtype=float)
        for i in vertices.shape[0]:
            vertices_complete[i] = vertices[i]

def find_bounds(vor, ivers, min_bounds=None, max_bounds=None):
    ivers_ = [i for i in ivers if i != -1]
    if not ivers_:
        return None
    vers = vor.vertices[ivers_]
    mins = vers.min(axis=0)
    maxs = vers.max(axis=0)
    bounds = np.vstack((mins, maxs)).T

    # consider the domain borders
    if min_bounds is None:
        min_bounds = (vor.points.min(axis=0)
            - np.abs(mins-vor.points.min(axis=0)))
        max_bounds = (vor.points.max(axis=0)
            + np.abs(maxs-vor.points.max(axis=0)))
    for i, bound in enumerate(bounds):
        if (bound[0]-bound[1]) == 0:
            if (bound[0]-min_bounds[i]) < (max_bounds[i]-bound[1]):
                bound[0] = min_bounds[i]
            else:
                bound[1] = max_bounds[i]
    
    return bounds

def get_point_bounds(points, min_bounds=None, max_bounds=None):
    '''Get the [min, max] boundaries for each voronoi cells
    Args:
        points (ndarray(np,ndim)): points
        min_bounds (ndarray(ndim)): optional. minimum value 
            for each dimension
        max_bounds (ndarray(ndim)): optional. maximum value
            for each dimension
    Returns:
        point_bounds (ndarray(np,ndim,2)): [min, max] boundaries for
            each dimension for each point (or voronoi cell)
    '''
    vor = Voronoi(points)
    ndim = vor.points.shape[1]

    point_bounds = np.zeros((len(points), ndim, 2), dtype=float)
    for ip, ireg in enumerate(vor.point_region):
        ivers = vor.regions[ireg]
        bounds = find_bounds(vor, ivers, min_bounds, max_bounds)
        point_bounds[ip] = bounds
    return point_bounds

def find_neighbour_regions(vor, ip):
    '''Find indices of voronoi cells (regions) neighbour to region
        of point ip
    Args:
        vor (Voronoi):
        ip (int): index of point within the target region
    Returns:
        iregs_neighbour (list(int)): indices of neighbour regions
    '''
    ireg_target = vor.point_region[ip]
    iverts_target = vor.regions[ireg_target]
    iverts_target = [i for i in iverts_target if i != -1]
    iregs_neighbour = []

    start_time = time.time_ns()

    for (ip1, ip2) in vor.ridge_points:
        if ip1 == ip:
            ireg = vor.point_region[ip2]
            iregs_neighbour.append(ireg)
        elif ip2 == ip:
            ireg = vor.point_region[ip1]
            iregs_neighbour.append(ireg)
    end_time = time.time_ns()
    
    return iregs_neighbour

def find_point_of_region(vor, ireg):
    '''Return the index of point within region ireg'''
    for ip, iregx in enumerate(vor.point_region):
        if iregx == ireg:
            return ip
    return None

def compute_distances_to_points(points, p_arr, ips=slice(None)):
    '''
    Args:
        vor (Voronoi):
        p_arr (ndarray): coordinates of anchor point
        ips (list(int)): indices of target points
    Returns:
        square of distances
    '''
    p_arr_ = p_arr.reshape(1,-1)

    if type(points) != np.ndarray:
        px_arr = np.zeros((len(ips), p_arr_.shape[1]))
        for i, ipx in enumerate(ips):
            px_arr[i] = points[ipx]
    else:
        px_arr = points[ips]

    res = px_arr - p_arr_
    dist2 = (res**2).sum(axis=1)

    return dist2

def find_bound_for_dim(
        vor, ip, idim, min_bound=None, max_bound=None,
        step_size=0.01, n_step_max=1000, log=None):
    '''Find the lower and upper boundaries of region of point ip
        along dimension idim
    Args:
        vor (Voronoi):
        ip (int): index of point within region of interest
        idim (int): the dimension along which to find the boundaries
        step_size (float): size of the increment to find
            the intersection
        n_step_max (int): maximum number of steps
    Returns:
        lower (float): distance to lower boundary
        upper (float): distance to upper boundary
    '''
    start_time = time.time_ns()
    iregs_neigh = find_neighbour_regions(vor, ip)
    end_time = time.time_ns()
    if log:
        log.write(
            'neighb. regions found in {} s\n'
            .format((end_time-start_time)*1e-9))

    ips_neigh = [find_point_of_region(vor, ireg) for ireg in iregs_neigh]
    
    # corrs_dim_neigh = np.array([vor.points[i,idim] for i in ips_neigh])

    # find distance to upper boundary
    p_arr = np.array(vor.points[ip])
    i = 0
    dist2 = compute_distances_to_points(vor.points, p_arr, ips_neigh)
    dist_to_ip = 0.
    start_time = time.time_ns()
    while (
            dist2.min() > dist_to_ip
            and i < n_step_max
            and p_arr[idim] < max_bound):
        i += 1
        p_arr[idim] += step_size
        dist2 = compute_distances_to_points(vor.points, p_arr, ips_neigh)
        dist_to_ip = (i*step_size)**2
    dist_up_bound = np.sqrt(dist_to_ip)
    end_time = time.time_ns()
    if log:
        log.write(
            'upper bound found in {} s\n'
            .format((end_time-start_time)*1e-9))

    # find distance to lower boundary
    p_arr = np.array(vor.points[ip])
    i = 0
    dist2 = compute_distances_to_points(vor.points, p_arr, ips_neigh)
    dist_to_ip = 0.
    while (
            dist2.min() > dist_to_ip
            and i < n_step_max
            and p_arr[idim] > min_bound):
        i += 1
        p_arr[idim] -= step_size
        dist2 = compute_distances_to_points(vor.points, p_arr, ips_neigh)
        dist_to_ip = (i*step_size)**2
    dist_lo_bound = -np.sqrt(dist_to_ip)

    return np.array([dist_lo_bound, dist_up_bound])

def implicit_find_bound_for_dim(
        points, ip, idim, n_nearest=10, min_bound=None, max_bound=None,
        step_size=0.01, n_step_max=1000, log=None):
    '''Without explicitely computing the voronoi diagram'''
    mask = np.ones(points.shape[0], bool)
    mask[ip] = False
    points_ = points[mask]

    dist2_0 = compute_distances_to_points(points_, points[ip])
    ips_neigh = np.argsort(dist2_0)[1:n_nearest+1]

    # find distance to upper boundary
    dist_to_ip = 0.
    i = 0
    dist2 = np.array(dist2_0)
    p_arr = np.array(points[ip])
    while (
            dist2.min() > dist_to_ip
            and i < n_step_max
            and p_arr[idim] < max_bound):
        i += 1
        p_arr[idim] += step_size
        dist2 = compute_distances_to_points(points_, p_arr, ips_neigh)
        dist_to_ip = (i*step_size)**2
    dist_up_bound = np.sqrt(dist_to_ip)
    if log:
        log.write(
            'upper bound found in {} s\n'
            .format((end_time-start_time)*1e-9))

    # find distance to lower boundary
    i = 0
    dist2 = np.array(dist2_0)
    dist_to_ip = 0.
    p_arr = np.array(points[ip])
    while (
            dist2.min() > dist_to_ip
            and i < n_step_max
            and p_arr[idim] > min_bound):
        i += 1
        p_arr[idim] -= step_size
        dist2 = compute_distances_to_points(points_, p_arr, ips_neigh)
        dist_to_ip = (i*step_size)**2
    dist_lo_bound = -np.sqrt(dist_to_ip)

    return np.array([dist_lo_bound, dist_up_bound])

if __name__ == '__main__':
    points = np.array(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
        [2, 0], [2, 1], [2, 2]])
    # min_bounds = np.array([0., 0.])
    # max_bounds = np.array([2.5, 2.5])

    # point_bounds = get_point_bounds(points, min_bounds, max_bounds)
    # print(point_bounds)

    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(40,12))

    start_time = time.time_ns()
    vor = Voronoi(points)
    end_time = time.time_ns()
    print(
        'Voronoi diag build in {} s'.format((end_time-start_time)*1e-9))

    # voronoi_plot_2d(vor)
    # plt.savefig('vor_example.pdf')

    ip = 20
    idim = 3

    # find bounds using the computed Voronoi diagram
    start_time = time.time_ns()
    ireg = vor.point_region[ip]
    iregs_neigh = find_neighbour_regions(vor, ip)
    ips_neigh = [find_point_of_region(vor, i) for i in iregs_neigh]

    up, lo = find_bound_for_dim(vor, ip, idim, -0.5, 2.5)
    print(up, lo)
    end_time = time.time_ns()
    print(
        'Explicit method: Bounds found in {} s'
        .format((end_time-start_time)*1e-9))

    # find bounds without explicitely computing the Voronoi diagram
    start_time = time.time_ns()
    up_imp, lo_imp = implicit_find_bound_for_dim(
        points, ip, idim, n_nearest=15, min_bound=-0.5, max_bound=2.5)
    print(up_imp, lo_imp)
    end_time = time.time_ns()
    print(
        'Implicit method: Bounds found in {} s'
        .format((end_time-start_time)*1e-9))

    # print('anchor point', vor.points[ip])
    # print('neighbour points', [vor.points[i] for i in ips_neigh])
    # print('up, low boundaries', up, lo)

