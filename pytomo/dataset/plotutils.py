import pygmt


def display_dataset(
        fig, stations, events, panel, proj='W15c',
        frame=["af", "WSne"]):
    """Plot stations and events on geographical map using PyGMT.

    Args:
        fig (Figure): matplotlib Figure object
        stations (list of Station):
        events (list of Event):
        panel (list of int): of format [irow, icol]
        proj (str): name of the geographical projection
        frame (list of str):

    """
    fig.basemap(
        region='g', projection=proj,
        panel=panel, frame=frame
    )
    fig.coast(shorelines='1/0.5p,black', projection=proj,
              panel=panel)
    fig.plot(
        x=stations.lon,
        y=stations.lat,
        sizes=[0.3]*len(stations),
        color='cyan',
        style="i",
        pen="black",
        panel=panel,
        projection=proj
    )
    fig.plot(
        x=events.lon,
        y=events.lat,
        sizes=[0.6]*len(events),
        color='red',
        style="a",
        pen="black",
        panel=panel,
        projection=proj
    )
