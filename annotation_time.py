import os
import numpy as np
import cv2
import pickle
from time import time
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from tqdm import tqdm



def read_image(path_cs, split, fname):
    fnima = os.path.join(path_cs, 'leftImg8bit', split, fname + '_leftImg8bit.png')
    ima = cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
    return ima

def read_masks(path_masks, split, fname):
    fnmasks = os.path.join(path_masks, split, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


suffix = '0.86_0.92_400'
min_area = 1000
path_cs = '/home/joans/Programs/mmsegmentation/data/cityscapes'
path_masks = 'masks_{}/cityscapes'.format(suffix)
split = 'train'

fname_all_masks = 'all_masks_cityscapes_{}_{}_{}.pkl'.format(split, suffix, min_area)
with open(fname_all_masks, 'rb') as f:
    all_masks = pickle.load(f)

num_total_masks = len(all_masks)
num_masks_to_annotate = 20
to_annotate = np.random.choice(num_total_masks, num_masks_to_annotate, replace=False)

legend = plt.imread('cityscapes_legend.png')

dict_classes = {0:'road', 1:'sidewalk', 2:'building', 3:'wall', 4:'fence', 5:'pole',
                6:'traffic light', 7:'traffic sign', 8:'vegetation', 9:'terrain',
                10:'sky', 11:'person', 12:'rider', 13:'car', 14:'truck', 15:'bus', 16:'train',
                17:'motorcycle', 18:'bicycle'}

to_show = []
classes = []
cverts = []
chorzs = []
for nm in tqdm(to_annotate):
    t1 = time()
    mask = all_masks[nm]
    fname = mask['fname']
    idx_mask = mask['nmask_in_image']
    masks_image = read_masks(path_masks, split, mask['fname'])
    seg = masks_image[idx_mask]['segmentation']
    cvert = np.argmax(np.sum(seg, axis=0))
    chorz = np.argmax(np.sum(seg, axis=1))
    ima = read_image(path_cs, split, fname)
    ima_with_boundary = mark_boundaries(ima, seg, color=(1, 0, 0), mode='thick')
    to_show.append(np.vstack([ima_with_boundary, legend]))
    classes.append(dict_classes[mask['label']])
    cverts.append(cvert)
    chorzs.append(chorz)


t_init = time()
for i in range(num_masks_to_annotate):
    ima_to_show = to_show[i]
    class_ = classes[i]
    cvert = cverts[i]
    chorz = chorzs[i]
    t1 = time()
    fig = plt.figure()
    plt.imshow(ima_to_show)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.text(20, 20, class_, color='white')
    plt.text(20, 60, class_, color='black')
    plt.plot([cvert, cvert], [0, 1023], 'r:')
    plt.plot([0, 2047], [chorz, chorz], 'r:')
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close(fig)
    t2 = time()
    print(t2-t1)

t_end = time()
total_time = t_end - t_init
print('total time {}, time per image {}'.format(total_time, total_time/num_masks_to_annotate))

