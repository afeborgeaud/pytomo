from pytomo.dataset.sourceselection import SourceSelection
from dsmpy.utils import cmtcatalog
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import datetime
import pickle
import glob

def _cat_depth(depth):
    # if depth <= 400:
    #     depth_cat = 0.
    # else:
    #     depth_cat = 1.
    depth_cat = depth // 100
    return depth_cat


def _get_X(catalog, use_depth=False):
    n_dim = 3 if use_depth else 2
    X = np.zeros((len(catalog), n_dim), dtype=np.float32)
    X_cat = np.zeros((len(catalog), n_dim), dtype=np.float32)
    r = 6371.
    for i, e in enumerate(catalog):
        x = np.radians(e.longitude + 180.) * r
        y = np.radians(e.latitude + 90) * r
        if use_depth:
            z_cat = _cat_depth(e.depth)
            z = e.depth
            X[i] = np.array([x, y, z])
            X_cat[i] = np.array([x, y, z_cat])
        else:
            X[i] = np.array([x, y])
            X_cat[i] = np.array([x, y])
    return X_cat, X


def cluster(catalog, max_clusters=50, max_dist=400):
    X_cat, X = _get_X(catalog)
    scaler = StandardScaler()
    scaler.fit(X_cat)
    X_scaled = scaler.transform(X_cat)
    if X_scaled.shape[1] == 3:
        X_scaled[:, 2] *= 2
    dists_max = []
    found = False
    if max_clusters > X_scaled.shape[0]:
        max_clusters = X_scaled.shape[0]
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        dist_max = 0.
        for j, label in enumerate(kmeans.labels_):
            dist = np.sqrt(np.dot(X[j, :2] - centers[label, :2],
                                  X[j, :2] - centers[label, :2]))
            if dist > dist_max:
                dist_max = dist
        dists_max.append(dist_max)
        if dist_max <= max_dist and not found:
            kmeans_best = kmeans
            found = True
    if not found:
        print('Maximum distance of {} not reached'.format(max_dist))
        kmeans_best = KMeans(n_clusters=1, random_state=0)
        kmeans_best.fit(X_scaled)

    centers = scaler.inverse_transform(kmeans_best.cluster_centers_)
    centers[:, :2] = np.degrees(centers[:, :2] / 6371.)
    centers[:, 0] -= 180.
    centers[:, 1] -= 90.

    return kmeans_best.labels_, centers


def get_dataframe(
        catalog, cluster_centers, cluster_labels, min_cluster_size=1):
    centers_expanded = [cluster_centers[label] for label in cluster_labels]
    lats = [e.latitude for e in catalog]
    lons = [e.longitude for e in catalog]
    deps = [e.depth for e in catalog]
    clon = [c[0] for c in centers_expanded]
    clat = [c[1] for c in centers_expanded]
    if (cluster_centers.shape[1] == 3):
        cdep = [1 if c[2] > 0.5 else 0 for c in centers_expanded]
    else:
        cdep = [0 for i in range(len(centers_expanded))]
    mws = [e.mt.Mw for e in catalog]
    data = dict(
        event=catalog, lat=lats, lon=lons, dep=deps, mw=mws,
        label=cluster_labels, clon=clon, clat=clat, cdep=cdep)
    df = pd.DataFrame(data)
    df.sort_values(by='label', inplace=True)
    df = _set_min_cluster_size(df, min_cluster_size)
    return df

def _set_min_cluster_size(df, size):
    counts = df['label'].value_counts(sort=False)
    large_counts = counts[counts >= size]
    large_df = df[df.label.isin(large_counts.index)]
    labels = large_df['label'].values.copy()
    for i in range(len(large_counts)):
        large_df.loc[labels==large_counts.index[i], 'label'] = i
    return large_df

def plot(
        df, lon_min=-180, lon_max=180, lat_min=90, lat_max=90,
        proj=ccrs.PlateCarree(), lon_0=-180, cluster_labels=None,
        cmap=None):
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    ax.coastlines(resolution='50m', color='black', linewidth=1.)
    gl = ax.gridlines(color='black', linewidth=.5, linestyle='--')
    gl.bottom_labels = True
    gl.left_labels = True
    ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    ax.add_feature(cartopy.feature.LAND, zorder=0, edgecolor='black')
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)


    lons = [event.longitude for event in df.event]
    lats = [event.latitude for event in df.event]
    if cluster_labels is None:
        plt.scatter(
            lons, lats, marker='*', color='r', s=5, zorder=10,
            transform=proj)
    else:
        if cmap is None:
            cmap = ListedColormap(sns.color_palette('bright', 10).as_hex())
            labels = np.mod(cluster_labels, 10)
        else:
            labels = cluster_labels
        plt.scatter(
            lons, lats, marker='*', c=labels, s=12, zorder=10,
            cmap=cmap, transform=proj)

        for i, row in df.drop_duplicates(subset=['label']).iterrows():
            ax.text(
                row.clon, row.clat, str(row.label),
                transform=proj, color='red', zorder=100)

    return fig, ax


def plot_cartesian(df, palette=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
    n_labels = len(df.label.unique())
    nh = min(df.label.max(), 5)
    ns = min(df.label.max(), 3)
    palette = sns.color_palette('bright', nh)
    styles = np.mod(df.label.values-1, ns)
    hues = np.mod(df.label.values-1, nh)
    sns.scatterplot(
        x='lon', y='lat', hue=hues, style=styles, data=df, ax=ax1,
        palette=palette, legend=None, s=10, edgecolor=None)
    sns.scatterplot(
        x='dep', y='lat', hue=hues, style=styles, data=df, ax=ax2,
        palette=palette, legend=None, s=10, edgecolor=None)
    ax1.set(xlabel='Longitude (deg)', ylabel='Latitude (deg)')
    ax2.set(xlabel='Depth (km)', ylabel='Latitude (deg)')

    return fig, (ax1, ax2)


def get_clusters_as_list(df):
    """Return a list of list of Event

    Args:
        df (DataFrame): see get_dataframe()

    Returns:
        list ot list of Event: list of event clusters

    """
    return df.groupby('label')['event'].apply(list).values.tolist()


def save_event_clusters(path, event_clusters):
    """Save the event clusters using pickle.dump().

        Args:
            path (str): name of the output file.
            event_clusters (list of list of Events): event clusters.

        """
    with open(path, 'wb') as f:
        pickle.dump(event_clusters, f)


def load_event_clusters(path):
    """Read event clusters using pickle.load().

    Args:
        path (str): name of the file that contains the clusters

    Returns:
        list of list of Event: event clusters.

    """
    with open(path, 'rb') as f:
        clusters = pickle.load(f)
    return clusters


if __name__ == '__main__':
    catalog = cmtcatalog.read_catalog()

    dep_min = 100
    dep_max = 800
    dist_min = 10
    dist_max = 35
    Mw_min = 5.3
    Mw_max = 7.
    start_date = datetime(2000, 1, 1)
    max_clusters = 50
    max_dist_in_km = 220.
    min_n_event = 2

    sac_files = glob.glob(
        '/mnt/doremi/anpan/inversion/MTZ_JAPAN/DATA/20*/*T')

    selector = SourceSelection(
        sac_files, dep_min=dep_min, dep_max=dep_max, Mw_min=Mw_min,
        Mw_max=Mw_max, dist_min=dist_min, dist_max=dist_max,
        start_date=start_date)
    catalog_filt = np.array(
        [event for event in selector.dataset.events
         if selector.select(event)])

    print('len(catalog)={}'.format(catalog_filt.shape))

    # exclude Philippines and Aleutians eqs
    catalog_filt = [e for e in catalog_filt
                    if not (e.longitude < 140 and e.latitude < 18)
                    and e.longitude < 165]

    cluster_labels, cluster_centers = selector.cluster(
        catalog_filt, max_clusters=max_clusters, max_dist=max_dist_in_km)

    _, counts = np.unique(cluster_labels, return_counts=True)
    cluster_centers_keep = cluster_centers[counts >= min_n_event]
    catalog_keep = [e for i, e in enumerate(catalog_filt)
                    if counts[cluster_labels[i]] >= min_n_event]
    cluster_labels_keep = [label for i, label in enumerate(cluster_labels)
                           if counts[cluster_labels[i]] >= min_n_event]

    print("n_events={}\nn_clusters={}"
          .format(len(catalog_keep), len(cluster_centers_keep)))

    df = selector.get_dataframe(
        catalog_keep, cluster_centers, cluster_labels_keep)

    selector.plot(
        catalog_keep, projection='cyl',
        lon_min=120, lon_max=160, lat_min=10, lat_max=60,
        cluster_labels=cluster_labels_keep)

    df.index = list(range(len(df)))
    df.to_csv('clusters.txt', sep=' ')
    selector.plot_cartesian(df)

    event_clusters = get_clusters_as_list(cluster_labels_keep, catalog_keep)
    save_event_clusters('event_cluster.pkl', event_clusters)
