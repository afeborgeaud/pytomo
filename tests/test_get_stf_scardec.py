import dsmpy.utils.scardec as scardec
from dsmpy.utils.cmtcatalog import read_catalog
from dsmpy.dataset import Dataset
import glob

if __name__ == '__main__':
    sac_files = glob.glob(
        '/mnt/doremi/anpan/inversion/MTZ_JAPAN/DATA/2*/*Z')
    dataset = Dataset.dataset_from_sac(sac_files)

    for event in dataset.events:
        print(event)
        duration_scardec = scardec.get_duration(event)
        if duration_scardec is None:
            print('Scardec STF not found for {}'.format(event))
        print(
            'scardec_duration, gcmt_duration:  {} {}'
            .format(duration_scardec,
                    2*event.source_time_function.half_duration))
