from pytomo.dataset.plotutils import display_windows
from dsmpy.windowmaker import WindowMaker
import pygmt

if __name__ == '__main__':
    windows = WindowMaker.load('selected_shift_windows_ScS_S_sS.pkl')
    windows = windows['0.01_0.08']

    windows_selected = [
        w for w in windows
        if (
            (-170 < w.station.longitude < -150
             or (-130 < w.station.longitude < -100
                and 30 < w.station.latitude < 40)
            )
            and (w.event.longitude > 175 or w.event.longitude < -170)
        )
    ]

    fig = pygmt.Figure()
    with fig.subplot(
            nrows=1,
            ncols=1,
            figsize=('20c', '20c'),
            frame=["af", "WSne"],
            # autolabel='(a)'
    ):
        display_windows(
            fig, windows_selected, ['ScS'], [0, 0], proj='G-160/30/18c',
            frame=["af", "WSne", "g"], region=[-100, 220, -60, 70]
        )
    fig.savefig('map.pdf')
