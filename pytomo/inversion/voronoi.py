import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

def compute_distances_to_points(points, p_arr, ips=slice(None)):
    """

    Args:
        points (ndarray):
        p_arr (ndarray): coordinates of anchor point
        ips (list of int): indices of target points
            (default is slice(None))

    Returns:
        ndarray: square of distances

    """
    p_arr_ = p_arr.reshape(1, -1)

    if type(points) != np.ndarray:
        px_arr = np.zeros((len(ips), p_arr_.shape[1]))
        for i, ipx in enumerate(ips):
            px_arr[i] = points[ipx]
    else:
        px_arr = points[ips]

    res = px_arr - p_arr_
    dist2 = (res ** 2).sum(axis=1)

    return dist2


def implicit_find_bound_for_dim(
        points, anchor_point, current_point,
        idim, n_nearest=10, min_bound=None,
        max_bound=None, step_size=0.01, n_step_max=1000, log=None):
    """Find the lower and upper boundaries of region of current_point
    along dimension idim, without explicitly computing
    the voronoi diagram, by performing a linear grid search.

    Args:
        points (ndarray): voronoi points
        anchor_point (ndarray): center of the voronoi cell
            of current_point.
        current_point (ndarray): point from which to compute distances
            to voronoi cell boundaries.
        idim (int): index of the dimension along which to compute the
            distance to boundaries.
        n_nearest (int): number of voronoi points used to compute the
            distance to boundaries.
        min_bound (float): minimum acceptable distance to boundaries.
            Used to avoid distances larger than largest perturbations
            defined in range_dict.
        max_bound (float): maximum acceptable distance.
        step_size (float): step size for the grid search
            (default is 0.01).
        n_step_max (int): maximum number of steps in the grid search
            (default is 1000).
        log (file): log file (default is None).

    Returns:
        ndarray: [lower, upper] bounds

    """
    mask = ~(points == anchor_point).all(axis=1)
    points_ = points[mask]

    dist2_0 = compute_distances_to_points(points_, current_point)
    ips_neigh = np.argsort(dist2_0)[:n_nearest]

    # find distance to upper boundary
    p_arr = np.array(current_point)
    dist_to_anch = np.dot(
        p_arr - anchor_point, p_arr - anchor_point)
    i = 0
    dist2 = np.array(dist2_0)
    while (
            dist2.min() > dist_to_anch
            and i < n_step_max
            and p_arr[idim] < max_bound):
        i += 1
        p_arr[idim] += step_size
        dist2 = compute_distances_to_points(points_, p_arr, ips_neigh)
        dist_to_anch = np.dot(
            p_arr - anchor_point, p_arr - anchor_point)

    if i > 0:
        p_arr[idim] -= step_size
    dist_to_current = np.dot(p_arr - current_point, p_arr - current_point)
    dist_up_bound = np.sqrt(dist_to_current)

    # find distance to lower boundary
    p_arr = np.array(current_point)
    dist_to_anch = np.dot(
        p_arr - anchor_point, p_arr - anchor_point)
    i = 0
    dist2 = np.array(dist2_0)
    while (
            dist2.min() > dist_to_anch
            and i < n_step_max
            and p_arr[idim] > min_bound):
        i += 1
        p_arr[idim] -= step_size
        dist2 = compute_distances_to_points(points_, p_arr, ips_neigh)
        dist_to_anch = np.dot(
            p_arr - anchor_point, p_arr - anchor_point)

    if i > 0:
        p_arr[idim] += step_size
    dist_to_current = np.dot(p_arr - current_point, p_arr - current_point)
    dist_lo_bound = -np.sqrt(dist_to_current)

    return np.array([dist_lo_bound, dist_up_bound])


def plot_voronoi_2d(
        points, misfits,
        xlim=None, ylim=None,
        ax=None, **kwargs):
    """Plot a voronoi diagram (only for 2-D points).

    Args:
        points (ndarray): voronoi points. (npoint, 2).
        misfits (ndarray): array of misfits for each voronoi point.
            (npoint,).
        xlim (list of float): x-axis range (default is None).
        ylim (list of float): y-axis range (default is None).
        ax (Axes): matplotlib Axes object (default is None).

    Returns:
        fig (figure): matplotlib figure object
        ax (Axes): matplotlib ax object
        scalar_map (): color map

    """
    assert points.shape[1] == 2
    # add dummy points
    # stackoverflow.com/questions/20515554/
    # colorize-voronoi-diagram?lq=1
    x_max = points[:, 0].max()
    x_min = points[:, 0].min()
    y_max = points[:, 1].max()
    y_min = points[:, 1].min()
    points_ = np.append(
        points,
        # [[x_min*10, y_min*10], [x_min*10, y_max*10],
        #  [x_max*10, y_min*10], [x_max*10, y_max*10]],
        [[-9999, -9999], [-9999, 9999], [9999, -9999], [9999, 9999]],
        axis=0)
    vor = Voronoi(points_)

    # color map
    # log_misfits = np.log(misfits)
    log_misfits = np.array(misfits)
    cm = plt.get_cmap('hot')
    c_norm = colors.Normalize(
        vmin=log_misfits.min(), vmax=log_misfits.max())
    # c_norm = colors.Normalize(vmin=0., vmax=0.3)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    voronoi_plot_2d(
        vor,
        show_vertices=False,
        line_colors='green',
        line_width=.5,
        show_points=False,
        ax=ax)

    ax.plot(
        points[:, 0], points[:, 1],
        '+g', markersize=5.)

    # colorize
    for ireg, reg in enumerate(vor.regions):
        if not -1 in reg:
            ip_ = find_point_of_region(vor, ireg)
            if ip_ == None:
                continue
            point = vor.points[ip_]
            ips = np.where(
                (points[:, 0] == point[0])
                & (points[:, 1] == point[1]))
            if ips[0].shape[0] > 0:
                ip = ips[0][0]
                color = scalar_map.to_rgba(log_misfits[ip])

                poly = [vor.vertices[i] for i in reg]
                ax.fill(*zip(*poly), color=color)

    ax.plot(0.2, 0., '*c', markersize=12)
    # ax.plot(0.2, -8./30., '*c', markersize=12)
    ax.set_aspect('equal')

    if xlim is None:
        xlim = [x_min, x_max]
        ylim = [y_min, y_max]
    ax.set(xlim=xlim, ylim=ylim)

    if ('title' in kwargs) and (fig is not None):
        fig.suptitle(kwargs['title'])

    return fig, ax, scalar_map


def find_point_of_region(vor, ireg):
    """Return the index of point within region ireg.

    Args:
        vor (Voronoi): Voronoi object.
        ireg (int): index of region.

    Returns:
        int: index of the Voronoi point for region ireg.

    """
    for ip, iregx in enumerate(vor.point_region):
        if iregx == ireg:
            return ip
    return None


if __name__ == '__main__':
    import time

    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(40, 4))
    misfits = rng.uniform(0., 1., size=(40, 1))

    start_time = time.time_ns()
    vor = Voronoi(points)
    end_time = time.time_ns()
    print(
        'Voronoi diag build in {} s'.format((end_time - start_time) * 1e-9))

    fig, ax, _ = plot_voronoi_2d(points, misfits)
    plt.savefig('vor_example.pdf')

    ip = 20
    idim = 3

    # find bounds without explicitly computing the Voronoi diagram
    start_time = time.time_ns()
    up_imp, lo_imp = implicit_find_bound_for_dim(
        points, points[ip], points[ip], idim,
        n_nearest=15, min_bound=-0.5, max_bound=2.5)
    print('up, low boundaries:', up_imp, lo_imp)
    end_time = time.time_ns()
    print(
        'Implicit method: Bounds found in {} s'
            .format((end_time - start_time) * 1e-9))
