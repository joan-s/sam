import os
import numpy as np
from tqdm import tqdm
import cv2
import pickle
import matplotlib.pyplot as plt
# from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
# from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.pools import ParallelPool
import time
from itertools import groupby


def read_gt(path_cs, split, fname):
    path_gt = os.path.join(path_cs, 'gtFine')
    fngt = os.path.join(path_gt, split, fname + '_gtFine_labelTrainIds.png')
    gt = cv2.imread(fngt, cv2.IMREAD_GRAYSCALE)
    return gt


def read_masks(path_masks, split, fname):
    fnmasks = os.path.join(path_masks, split, fname + '_masks.npz')
    return np.load(fnmasks, allow_pickle=True)['masks']


def check_all_masks_and_gt_exist(path_cs, path_masks, split, fnames):
    print('check all masks and gt exist...')
    path_gt = os.path.join(path_cs, 'gtFine')
    for fname in tqdm(fnames):
        fnmasks = os.path.join(path_masks, split, fname + '_masks.npz')
        assert os.path.isfile(fnmasks), fnmasks + ' doesn''t exist'
        fngt = os.path.join(path_gt, split, fname + '_gtFine_labelTrainIds.png')
        assert os.path.isfile(fngt)
    print('done')


def compute_all_masks_fn(path_cs, path_masks, split, num_classes, fn, min_area):  # 439347 masks in 1/2 hour, min_area=1000
    all_masks_fn = []
    masks = read_masks(path_masks, split, fn)
    gt = read_gt(path_cs, split, fn)
    num_masks_ima = len(masks)
    for nmi in range(num_masks_ima):
        area = masks[nmi]['area']
        if area > min_area:
            seg = masks[nmi]['segmentation']
            counts_gt = np.bincount(gt[seg], minlength=num_classes) # more if 255's
            majority_label = np.argmax(counts_gt)
            if majority_label != 255:
                # this is to discard the void regions in the groundtruth like the own car motor cover
                all_masks_fn.append({'fname': fn,
                                     'nmask_in_image': nmi,
                                     'area': area,
                                     'counts_gt': counts_gt[:num_classes], # don't count 255's
                                     'label': majority_label,
                                     'annotated': False})
    print('.', end='', flush=True)
    return all_masks_fn


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def compute_all_masks(path_cs, path_masks, split, num_classes, fnames, min_area):  #
    all_masks = []
    num_cores = cpu_count()
    with ThreadPool(processes=num_cores) as pool:
        res = pool.starmap(compute_all_masks_fn,
                           [(path_cs, path_masks, split, num_classes, fn, min_area)
                            for fn in fnames])
        all_masks.extend(res)

    all_masks = flatten_list(all_masks)
    print('all_masks has {} masks'.format(len(all_masks)))
    return all_masks


def make_all_masks(path_cs, path_masks, split, num_classes, suffix, min_area, seed):
    np.random.seed(seed)
    list_fnames = os.path.join(path_cs, '{}.txt'.format(split))
    with open(list_fnames, 'r') as f:
        fnames = f.read().splitlines()

    check_all_masks_and_gt_exist(path_cs, path_masks, split, fnames)
    # it's a long process and before starting we make sure all files are there

    print('computing all masks...')
    t1 = time.time()
    all_masks = compute_all_masks(path_cs, path_masks, split, num_classes, fnames, min_area)
    t2 = time.time()
    print('{} masks in {} seconds'.format(len(all_masks), t2 - t1))
    print('done')

    return all_masks


if __name__ == '__main__':
    suffix = '0.86_0.92_400'
    split = 'train'
    min_area = 1000
    rows, cols = 1024, 2048
    num_classes = 19 # excluding 255 = unknown

    path_cs = '/home/joans/Programs/mmsegmentation/data/cityscapes'
    path_masks = 'masks_{}/cityscapes'.format(suffix)
    # masks for each image computed before with save_masks_cityscapes.py
    seed = 1234
    all_masks = make_all_masks(path_cs, path_masks, split, num_classes, suffix, min_area, seed)
    # 439347 masks in 1724 seconds
    fname_all_masks = 'all_masks_cityscapes_{}_{}_{}.pkl'.format(split, suffix, min_area)
    # has all the info of all the saved masks but for the segmentation field
    with open(fname_all_masks, 'wb') as f:
        pickle.dump(all_masks, f)
