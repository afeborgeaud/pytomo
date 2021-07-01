import pygmt
import os

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
        x=[s.longitude for s in stations],
        y=[s.latitude for s in stations],
        sizes=[0.3]*len(stations),
        color='cyan',
        style="i",
        pen="black",
        panel=panel,
        projection=proj
    )
    fig.plot(
        x=[e.longitude for e in events],
        y=[e.latitude for e in events],
        sizes=[0.6]*len(events),
        color='red',
        style="a",
        pen="black",
        panel=panel,
        projection=proj
    )


def display_windows(
        fig, windows, phases, panel, proj='W15c',
        frame=["af", "WSne"], region='g'):
    """Plot stations, events, and ray paths
    on geographical map using PyGMT.

    Args:
        fig (Figure): matplotlib Figure object
        windows (list of Window): time windows
        phases (list of str): seismic phases to plot
        panel (list of int): of format [irow, icol]
        proj (str): name of the geographical projection
        frame (list of str):

    """
    stations = []
    events = []
    for window in windows:
        if window.station not in stations:
            stations.append(window.station)
        if window.event not in events:
            events.append(window.event)

    fig.basemap(
        region=region, projection=proj,
        panel=panel, frame=frame
    )

    # generate text data (much faster to plot)
    data = "\n>\n".join(
        [
            f'{w.event.longitude} {w.event.latitude}\n'
            f'{w.station.longitude} {w.station.latitude}'
            for w in windows
        ]
    )
    datapath = f'tmp.txt'
    with open(datapath, 'w') as f:
        f.write(data)
    fig.plot(
        data=datapath,
        pen=f'0.5p,orange',
        panel=panel,
        projection=proj,
        region=region
    )
    os.remove(datapath)

    fig.coast(shorelines='1/0.5p,black', projection=proj,
              panel=panel, region=region)
    fig.plot(
        x=[s.longitude for s in stations],
        y=[s.latitude for s in stations],
        sizes=[0.3]*len(stations),
        color='royalblue',
        style="i",
        pen="black",
        panel=panel,
        projection=proj,
        region=region
    )
    fig.plot(
        x=[e.longitude for e in events],
        y=[e.latitude for e in events],
        sizes=[0.6]*len(events),
        color='red',
        style="a",
        pen="black",
        panel=panel,
        projection=proj,
        region=region
    )
