from pydsm.utils import cmtcatalog
from pytomo.dataset.sourceselection import SourceSelection
import numpy as np
from datetime import datetime
import glob
import sys


if __name__ == '__main__':
    catalog = cmtcatalog.read_catalog()

    dep_min = 150
    dep_max = 800
    dist_min = 15
    dist_max = 25
    Mw_min = 5.3
    Mw_max = 7.
    start_date = datetime(2000,1,1)

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
        catalog_filt, max_clusters=50, max_dist=220)

    _, counts = np.unique(cluster_labels, return_counts=True)
    cluster_centers_keep = cluster_centers[counts >= 2]
    catalog_keep = [e for i,e in enumerate(catalog_filt)
                    if counts[cluster_labels[i]] >= 2]
    cluster_labels_keep = [label for i,label in enumerate(cluster_labels)
                            if counts[cluster_labels[i]] >= 2]

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