from pytomo.dataset.sourceselection import SourceSelection
import pytomo.dataset.sourcecluster as sc
from dsmpy.utils import cmtcatalog
import numpy as np
from datetime import datetime
import glob


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
    min_cluster_size = 2

    sac_files = glob.glob(
        '/mnt/ntfs/anselme/work/japan/DATA/2*/*T')

    selector = SourceSelection(
        sac_files, dep_min=dep_min, dep_max=dep_max, Mw_min=Mw_min,
        Mw_max=Mw_max, dist_min=dist_min, dist_max=dist_max,
        start_date=start_date)
    catalog_filt = np.array(
        [event for event in selector.dataset.events
         if selector.select(event)])

    print('len(catalog)={}'.format(catalog_filt.shape))

    # exclude Philippines and Aleutians events
    catalog_filt = [e for e in catalog_filt
                    if not (e.longitude < 140 and e.latitude < 18)
                    and e.longitude < 165]

    cluster_labels, cluster_centers = sc.cluster(
        catalog_filt, max_clusters=max_clusters, max_dist=max_dist_in_km)

    df = sc.get_dataframe(
        catalog, cluster_centers, cluster_labels, min_cluster_size)

    print("n_events={}\nn_clusters={}"
          .format(len(df), len(df.label.nunique())))

    sc.plot(
       df.event, projection='cyl',
       lon_min=120, lon_max=160, lat_min=10, lat_max=60,
       cluster_labels=df.label)

    df.index = list(range(len(df)))
    df.to_csv('clusters.txt', sep=' ')
    sc.plot_cartesian(df)

    event_clusters = df.groupby('label')['event'].apply(list)
    sc.save_event_clusters('event_cluster.pkl', event_clusters)
