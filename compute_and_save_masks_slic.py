import os
from tqdm import tqdm
import cv2
import argparse
import numpy as np
from skimage.segmentation import slic
from skimage.measure import label, regionprops
from multiprocessing.pool import Pool

"""
python -i compute_and_save_masks_slic.py \
  --n-segments 200 \
  --filename cityscapes_train.txt \
  --dataset cityscapes \
  --path-out masks_slic/cityscapes/train

python -i compute_and_save_masks_slic.py \
  --filename mapillary_14716_train_images_aspect_1.33.txt \
  --dataset mapillary_vistas_aspect_1.33 \
  --path-out masks_slic/Mapillary_Vistas_aspect_1.33/training 
  
python -i compute_and_save_masks_slic.py \
  --n-segments 1000 \
  --min-mask-region-area 100 \
  --filename easyportrait_train_1.txt \
  --dataset easyportrait \
  --path-out masks_slic/easyportrait/train  
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Computes and saves SLIC masks for all the images specified in a file.')
    parser.add_argument('--n-segments', type=int, default=200)
    parser.add_argument('--filename', required=True,
                        help='text file with the names of the images to be processed')
    parser.add_argument('--dataset', required=True,
                        help='either cityscapes of mapillary, this is to decide the format of the masks filenames '
                        'and if to resize the image')
    parser.add_argument('--min-mask-region-area', type=int, default=1000,
                        help='TODO')
    parser.add_argument('--path-out', required=True,
                        help='directory where the binary map for the SLIC regions of each image will be saved, '
                             'one file with all the binary maps per image')
    args = parser.parse_args()
    return args


def compute_slic_masks(ima):
    s = slic(ima, n_segments=args.n_segments)
    assert s.max() > 1
    masks = []
    for lab in range(1, s.max()):
        mask = s == lab
        area = mask.sum()
        assert area > 0
        if area > args.min_mask_region_area :
          masks.append({'segmentation': mask, 'area': area})
    return masks


def read_image(fn):
    image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
    if args.dataset == 'cityscapes':
        pass
    elif args.dataset == 'mapillary':
        image = cv2.resize(image, (1632, 1216))
    elif args.dataset == 'easyportrait':
        pass
        #image = image[::2, ::2, :]
    return image


def make_fname_mask(fn):
    if args.dataset == 'cityscapes':
        fname_mask = fn.split('/')[-1].replace('leftImg8bit', 'masks').replace('.png', '.npz')
        # to be read like
        # masks = np.load(fname_ima.replace('leftImg8bit', 'masks').replace('.png', '_masks.npz'),
        #                 allow_pickle=True)['masks_0.86_0.92_400']
    elif args.dataset == 'mapillary':
        fname_mask = fn.split('/')[-1].replace('.jpg', '.npz')
    elif args.dataset == 'easyportrait':
        _, _, split, fname_image = fn.split('/')[-4:]
        fname_mask = fname_image.replace('.jpg', '.npz')
    else:
        assert False

    return fname_mask


def process_one_image(fn):
    if args.dataset == 'easyportrait':
        fn = '/home/joans/109/datasets/easyportrait/images/train/' + fn

    image = read_image(fn)
    masks = compute_slic_masks(image)
    fname_mask = make_fname_mask(fn)
    print(fn, len(masks), 'masks')
    np.savez_compressed(os.path.join(args.path_out, fname_mask), masks=masks)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.filename)
    assert args.dataset in ['cityscapes', 'mapillary_vistas_aspect_1.33', 'easyportrait']
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)


    with open(args.filename, 'r') as f:
        fnames = f.read().splitlines()

    with Pool(processes=4) as pool:
        pool.map(process_one_image, fnames)





            
