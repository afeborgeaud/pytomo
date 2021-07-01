from pytomo.dataset.plotutils import display_dataset
from dsmpy.windowmaker import WindowMaker
import pygmt

if __name__ == '__main__':
    windows = WindowMaker.load('windows.pkl')
    stations = list(set([w.station for w in windows]))
    events = list(set([w.event for w in windows]))

    fig = pygmt.Figure()
    with fig.subplot(
            nrows=1,
            ncols=1,
            figsize=('32c', '29.5c'),
            frame=["af", "WSne"],
            autolabel='(a)'
    ):
        display_dataset(
            fig, stations, events, [0, 0], proj='W15c',
            format=["af", "WSne"]
        )
    fig.savefig('map.pdf')