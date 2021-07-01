from pytomo.dataset.plotutils import display_dataset
from dsmpy.windowmaker import WindowMaker
import pygmt

if __name__ == '__main__':
    windows = WindowMaker.load('selected_shift_windows_ScS_S_sS.pkl')
    windows = windows['0.01_0.08']
    stations = events = []
    for window in windows:
        if window.station not in stations:
            stations.append(window.station)
        if window.event not in events:
            events.append(window.event)

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
            frame=["af", "WSne"]
        )
    fig.savefig('map.pdf')
