import numpy as np

fnames = ['annotation_times_cityscapes.dat', 'annotation_times_mapillary.dat']
for fn in fnames:
    with open(fn,'r') as f:
        line = f.read().splitlines()

    assert len(line) == 1
    line = line[0]
    line = line.split(' ')
    times =  [float(str_time) for str_time in line if len(str_time)>0]
    print('{} times in {}'.format(len(times), fn))
    print('mean {}, std {}'.format(np.mean(times), np.std(times)))