from pydsm.utils import cmtcatalog
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns

def select(
        event, lat_min=-90, lat_max=90, lon_min=-180, lon_max=180,
        dep_min=0, dep_max=800, Mw_min=0, Mw_max=10,
        target_lats=None, target_lons=None, dist_min=0, dist_max=180,
        start_date=datetime(1970,1,1), end_year=2050):
    selected = False
    if event.event_id[-1] != 'A':
        return False
    if target_lats != None and target_lons != None:
        if (event.depth >= dep_min and event.depth <= dep_max
        and event.mt.Mw >= Mw_min and event.mt.Mw <= Mw_max
        and event.centroid_time >= start_date):
            distances = [event.get_epicentral_distance_(lat, lon)
                        for lat, lon in zip(target_lats, target_lons)]
            mask = [(d >= dist_min and d <= dist_max) for d in distances]
            selected = np.array(mask).any()
    elif (event.latitude >= lat_min and event.latitude <= lat_max
        and event.longitude >= lon_min and event.longitude <= lon_max
        and event.depth >= dep_min and event.depth <= dep_max
        and event.mt.Mw >= Mw_min and event.mt.Mw <= Mw_max
        and event.centroid_time >= start_date):
        selected = True
    return selected

def plot(
    catalog, lon_min=-180, lon_max=180, lat_min=90, lat_max=90,
    projection='robin', lon_0=-180, cluster_labels=None,
    cmap=None):
    fig, ax = plt.subplots()

    m = Basemap(resolution='i', projection=projection, lon_0=lon_0,
                llcrnrlon=lon_min, urcrnrlon=lon_max,
                llcrnrlat=lat_min, urcrnrlat=lat_max, ax=ax)
    m.drawcoastlines()
    m.drawmapboundary(fill_color=None)
    m.fillcontinents(color='wheat', lake_color='white')

    m.drawparallels(
        np.arange(-80., 81., 20.),labels=[True, False, False, False],
        labelstyle='+/-', fontsize=10)
    m.drawmeridians(
        np.arange(-180., 181., 20.), labels=[False, False, False, True],
        labelstyle='+/-', fontsize=10)

    lons = [event.longitude for event in catalog]
    lats = [event.latitude for event in catalog]
    x_eq, y_eq = m(lons, lats)
    if cluster_labels is None:
        m.scatter(x_eq, y_eq, marker='*', color='r', s=5, zorder=10)
    else:
        if cmap is None:
            cmap = ListedColormap(sns.color_palette('bright', 10).as_hex())
            labels = np.mod(cluster_labels, 10)
        else:
            labels = cluster_labels
        m.scatter(
            x_eq, y_eq, marker='*', c=labels, s=12, zorder=10,
            cmap=cmap)
    # m.drawcoastlines(zorder=10)

    plt.savefig('map.pdf', bbox_inches='tight')

def _cat_depth(depth):
    # if depth <= 400:
    #     depth_cat = 0.
    # else:
    #     depth_cat = 1.
    depth_cat = depth // 100
    return depth_cat

def _get_X(catalog):
    X = np.zeros((len(catalog),3), dtype=np.float32)
    X_cat = np.zeros((len(catalog),3), dtype=np.float32)
    r = 6371
    for i, e in enumerate(catalog):
        x = np.radians(e.longitude + 180.) * r
        y = np.radians(e.latitude + 90) * r
        z_cat = _cat_depth(e.depth)
        z = e.depth
        X[i] = np.array([x, y, z])
        X_cat[i] = np.array([x, y, z_cat])
    return X_cat, X

def cluster(catalog, max_clusters=50, max_dist=400):
    X_cat, X = _get_X(catalog)
    scaler = StandardScaler()
    scaler.fit(X_cat)
    X_scaled = scaler.transform(X_cat)
    X_scaled[:,2] *= 2
    dists_max = []
    found = False
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        dist_max = 0.
        for j, label in enumerate(kmeans.labels_):
            dist = np.sqrt(np.dot(X[j,:2]-centers[label,:2],
                           X[j,:2]-centers[label,:2]))
            if dist > dist_max:
                dist_max = dist
        dists_max.append(dist_max)
        if dist_max <= max_dist and not found:
            kmeans_best = kmeans
            found = True
    if not found:
        print('Maximum distance of {} not reached'.format(max_dist))
        kmeans_best = kmeans
    # plt.plot(list(range(1, max_clusters)), dists_max)
    # plt.show()
    centers = scaler.inverse_transform(kmeans_best.cluster_centers_)
    centers[:,:2] = np.degrees(centers[:,:2] / 2. / 6371.)

    return kmeans_best.labels_, centers

def get_dataframe(catalog, cluster_centers, cluster_labels):
    centers_expanded = [cluster_centers[label] for label in cluster_labels]
    lats = [e.latitude for e in catalog]
    lons = [e.longitude for e in catalog]
    deps = [e.depth for e in catalog]
    clon = [c[0] for c in centers_expanded]
    clat = [c[1] for c in centers_expanded]
    cdep = [1 if c[2] > 0.5 else 0 for c in centers_expanded]
    ev_ids = [e.event_id for e in catalog]
    data = dict(
        id=ev_ids, lat=lats, lon=lons, dep=deps,
        label=cluster_labels, clon=clon, clat=clat, cdep=cdep)
    df = pd.DataFrame(data)
    df.sort_values(by='label', inplace=True)
    return df

def plot_cartesian(df, palette=sns.color_palette('bright', 10)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
    n_labels = len(df.label.unique())
    styles = np.mod(df.label.values, 8)
    hues = np.mod(df.label.values, 10)
    sns.scatterplot(
        x='lon', y='lat', hue=hues, style=styles, data=df, ax=ax1,
        palette=palette, legend=None, s=10, edgecolor=None)
    sns.scatterplot(
        x='dep', y='lat', hue=hues, style=styles, data=df, ax=ax2,
        palette=palette, legend=None, s=10, edgecolor=None)
    ax1.set(xlabel='Longitude (deg)', ylabel='Latitude (deg)')
    ax2.set(xlabel='Depth (km)', ylabel='Latitude (deg)')
    plt.savefig('clusters_cart.pdf', bbox_inches='tight')

        
if __name__ == '__main__':
    catalog = cmtcatalog.read_catalog()
    
    target_lats = [32.5, 36.0, 43.2]
    target_lons = [131.0, 138.5, 142.5]
    dep_min = 100
    dep_max = 800
    dist_min = 15
    dist_max = 25
    Mw_min = 5.5
    Mw_max = 7.
    start_date = datetime(2000,1,1)

    selector = partial(
        select, target_lats=target_lats, target_lons=target_lons,
        dep_min=dep_min, dep_max=dep_max,
        dist_min=dist_min, dist_max=dist_max, Mw_min=Mw_min, Mw_max=Mw_max,
        start_date=start_date)
    catalog_filt = np.array([event for event in catalog if selector(event)])
    
    # exclude Philippines and Aleutians eqs
    catalog_filt = [e for e in catalog_filt
                    if not (e.longitude < 140 and e.latitude < 18)
                    and e.longitude < 165]

    cluster_labels, cluster_centers = cluster(
        catalog_filt, max_clusters=50, max_dist=220)

    _, counts = np.unique(cluster_labels, return_counts=True)
    cluster_centers_keep = cluster_centers[counts >= 2]
    catalog_keep = [e for i,e in enumerate(catalog_filt)
                    if counts[cluster_labels[i]] >= 2]
    cluster_labels_keep = [label for i,label in enumerate(cluster_labels)
                           if counts[cluster_labels[i]] >= 2]

    print("n_events={}\nn_clusters={}"
          .format(len(catalog_keep), len(cluster_centers_keep)))

    df = get_dataframe(
        catalog_keep, cluster_centers, cluster_labels_keep)

    plot(
        catalog_keep, projection='cyl',
        lon_min=120, lon_max=160, lat_min=10, lat_max=60,
        cluster_labels=cluster_labels_keep)

    plot_cartesian(df)
    df.index = list(range(len(df)))
    df.to_csv('clusters.txt', sep=' ')
