from dsmpy.dataset import Dataset
import numpy as np
from datetime import datetime

class SourceSelection:
    """Select seismic events to use for inversion
    """

    def __init__(
            self, sac_files, lat_min=-90, lat_max=90,
            lon_min=-180, lon_max=180, dep_min=0, dep_max=800,
            Mw_min=0, Mw_max=10, dist_min=0,
            dist_max=180, n_within_dist=10,
            start_date=datetime(1970, 1, 1),
            end_date=datetime(2050, 1, 1)):
        self.sac_files = sac_files
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.dep_min = dep_min
        self.dep_max = dep_max
        self.Mw_min = Mw_min
        self.Mw_max = Mw_max
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.n_within_dist = n_within_dist
        self.start_date = start_date
        self.end_date = end_date

        self.dataset = Dataset.dataset_from_sac(
            sac_files, headonly=True)

    def select(self, event):
        """Check if the event satisfies the selection criteria

        Args:
            event (Event): seismic event.

        Returns:
            bool: true if the event satisfies the selection criteria.

        """
        iev = np.argwhere(self.dataset.events == event)[0][0]
        start, end = self.dataset.get_bounds_from_event_index(iev)
        stations = self.dataset.stations[start:end]

        if (event.event_id[-1] != 'A'
                or event.latitude < self.lat_min
                or event.latitude > self.lat_max
                or event.longitude < self.lon_min
                or event.longitude > self.lon_max
                or event.depth < self.dep_min
                or event.depth > self.dep_max
                or event.mt.Mw < self.Mw_min
                or event.mt.Mw > self.Mw_max
                or event.centroid_time < self.start_date
                or event.centroid_time > self.end_date):
            return False
        distances = np.array([event.get_epicentral_distance(station)
                              for station in stations])
        n_within = (
            (distances >= self.dist_min) & (distances <= self.dist_max)).sum()
        if n_within < self.n_within_dist:
            return False
        return True
