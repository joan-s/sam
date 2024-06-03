import os
import numpy as np
import cv2
import pickle
from time import time
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from matplotlib.backend_bases import MouseButton
from itertools import groupby
from tqdm import tqdm
import argparse
from metric import ConfusionMatrix
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

"""
(sam) joans@cvc178:~/Programs/sam$ python  annotation_time.py --num-masks-to-annotate-per-class 20 
region labels [16  2 13 15  9  7 16  1 14 13 14  4  3  2 16  6 10  9  2  6 17  7 14  1
  7 15  7  0 18 18 10 13  1  6  1  1  8  8  0  0  2 13 16  8  7 15 18 12
  0 17  2  3 13 16 15 16 10 15 16  2 10  7 15 14  1 17  2 15 10  9 14  6
  5  7  4  2  4  6  6 10  7  0  1  1  6  6  2  2  6  1 15  3 11  4  7 18
 11 12 14 11  8 18 11 15  7 10  6 11 11 13 11  5 10 13  2 12 18 15  0 13
 18 18  7 18  8 16 17 11  9 14  0 11  4  4 12 14 12 13  7  3  4 10  5  6
 10 18 13  1  5 12 14  9 12 15 15 16 14  6 18  8  1 15  9  4 16  0  9 11
 10  8 18  7  1  9  8  8 15  9  5  5  4 12 11 16  9 13  4 17  2 13 17 17
 16  9  2  5 17 10 16  1  9  2 10  0 17 15  2  8  3 16 11 18  1  4 16 14
  3  9 13  5 16  3 16 13  2  2 11  3  7  3  2 14 15 12  7  6 17  5 15  4
 17  0 10 17  6 18 18 10 18  8  5 12  3 13  3 14  5 18 15 12 11  3  1  5
  6  5  4  8  6 10  5 17 17  0  0 13  9 14  3 14 12  3  3  9 11  1 17  1
  4  5  0  7 17  6 16  8 11  0 12 13  2 12  3 17 12  0  6  3 13  4 11 10
  8 12  1 18 17  8 15  4  0 14 14 17  5 10  9  7  2  8  4  5  8 11  5  9
 17 13  9 10 14  8 12 10 13  5  9  0 11  0  8  7  0  3  1  4 12 14  3  3
  0 15  1 16 14 16 11 18 12  4  4  6  6  7 18  8  5 12  9  7]
annotation labels [16  2 13 15  9  7 16  1 14 13 14  4  3  2 16  7 10  9  2  6 17  7 14  1
  7 15  7  0 18 18 10 13  1  6  1  1  2  8  0  0  2 13 16  8  7 15 18 18
  0 17  2  3 13 16 15 16 10 15 16  2 10  7 15 14  1 17  2 15 10 16 14  6
  5  7  4  2  4  6  6 10  7  0  1  1  6  6  2  2  6  1 15  3 11  4  7 18
 11 12 14 11  8 18 11 15  7 10  6 11 11 13 11  5 10 13  2 12 18 15  0 13
 18 18  7 18  8 16 17 11  9 14  0 11  4  4 12 14 12 13  7  3  4 10  5  6
 10 18 13  1  5 11 14  9 12 15 15 16 14  6 18  8  1 15  9  4 16  0  9 11
 10  8 18  7  1  9  8  8 15  9  5  5  4 12 11 16  9 13  4 17  2 13 17 17
 16  9  2  5 17 10 16  1  9  2  5  0 17 15  2  8  3 16 13 18  1  4 16 14
  3  9 13  5 16  3 16 13  2  2 11  3  7  3  2 14 15 12  7  6 17  5 15  4
 17  0 10 17  6 18 18 10 18  8  5 12  3 13  3 14  5 18 15 12 11  3  1  5
  6  5  4  8  6 10  5  8 17  0  0 13  9 14  3 14 12  3  5  9 11  1 17  1
  4  5  0  7 17  6 16  2 11  0 12 13  2 12  3 17 12  0  6  3 13  4 11 10
  2 12  1 18 17  8 15  4  0 14 14 17  5 10  9  7  2  8  4  5  8 11  5  9
 17 13  9 10 14  8 12 10 13  5  9  0 11  0  8  7  0  3  1  4 12 14  3  3
  0 15  1 16 14 16 11 17 12  4  4  6  6  7 18  8  5 12  9  7]
annotation times [ 3.48219538  3.89269805  2.91089511  2.73724222  3.99525237  3.09236169
  2.5658958   4.24103856  4.79762387  3.86237907  4.88181925  6.21368146
  6.13914633  2.6552825   3.2050128   3.11941981  3.62584305  3.87474632
  3.33410096  3.87565875  2.56254458  2.44466972  3.63331676  2.65208244
  4.90891385  4.64167285  2.78618073  2.99230623  3.41942978  3.1231482
  3.98423052  2.55632567  3.99490738  4.49058938  3.6521399   3.78519726
  3.80582619  3.77107787  3.85251594  1.97427773  3.25686169  3.34990883
  3.39157462  3.49092627  4.80889893  2.87757087  4.79354787  2.88023138
  3.92068887  3.60194707  2.62752628  3.36664867  2.74454594  3.61146712
  2.30993843  4.07887125  3.82065034  2.7670579   2.60156798  4.51602769
  4.02160192  4.91539478  3.53898263  2.48245358  3.71627283  4.16253376
  3.13154626  2.84681964  2.96550417  5.43319058  6.32615995  3.62722158
  3.33747387  2.45779204  9.5910027   3.52344608  3.37901282  4.2631712
  2.76916909  2.57835865  2.91463757  2.47685051  3.53252029  3.51186562
  3.4354291   2.13859057  2.72707605  2.90485477  3.47995114  4.58242846
  2.96172214  4.5060451   3.14074588  4.11564994  4.17120957  4.90245748
  2.67559028 10.30951095  3.73164797  3.96842909  3.02682424  4.46775579
  8.18199992  2.69189668  4.07527542  2.96139884  3.65024614  2.39107585
  2.07864857  2.60622787  2.84299946  3.56298208  2.95680332  2.53889847
  3.43460155  4.42849445  3.28123569  3.19120002  3.59253621  2.29721498
  3.06259799  1.55466223  2.60775757  4.93557143  4.99086285  2.63687491
  3.74665833  3.04583049  4.22391081  4.25282741  3.78801012  2.33583331
  4.72726297  4.68208289  5.15497565  2.98551989  2.92027736  2.83750868
  2.94900346  4.84204149  3.81454849  2.59266281  5.65416169  4.31385708
  2.45998144  5.21043038  3.10052276  3.75189972  4.38126302  2.79910612
  4.83090639  4.22614908  4.77218676  3.19820189  3.27710295  2.73500419
  3.5913372   3.25853896  3.2909224   4.32380199  4.26550126  2.35429811
  4.1339426   4.27806783  3.17535663  3.60241032  4.48842692  2.95103359
  2.98716521  5.79755902  3.00287557  3.8031497   3.3664453   2.88906622
  3.58408499  2.77007079  4.62867498  4.80474091  3.92317009  3.66560459
  3.41921544  3.90888476  3.41248608  2.8850584   3.52878571  3.21058249
  4.11444926  3.22682524  4.42139292  2.31669617  2.85895324  2.37503576
  2.95087171  5.56855106  3.37912226  2.98407292  3.42520571  2.77406883
  3.12903714  2.92295551  3.41763926  4.20888686  9.94902873  2.820997
  2.49359941  2.56750536  7.38637114  3.77258182  4.32707047  3.07871056
  9.03888559  4.62241364  3.96381259  6.58212876  2.6403985   8.7900629
  4.38498783  4.64936042  2.55807614  3.07705688  2.0096333   2.348665
  3.74068356  2.42985177  4.16035771  2.69181752  2.73975277  3.5781703
  3.08944297  4.87096715  3.75773335  2.5663836   6.96066785  4.755898
  3.57513762  5.90865088  2.09797835  2.80123878  5.25565886  4.36121082
  3.27275491  3.32136226  3.66196918  3.16064906  3.78967714  3.14902616
  2.94543743  2.31824255  4.46090865  3.25267839  3.23180199  2.78648376
  5.07488155  3.75395751  5.06018972  3.00348592  3.55155206  8.06521916
  2.68560171  3.68500066  1.99979997  2.91183257  3.24212885  2.95378304
  3.20705938  3.09368134  3.72645688  3.51444244  3.11652589  3.03126192
  2.55145311  3.24177384  3.86936283  2.49798799  2.67248702  3.62655187
  4.29181314  8.48543978  2.72889948  3.41008186  3.52570987  4.73663449
  2.87985969  4.15446091  2.50705385  2.69434714  6.78960085  3.45496798
  4.4026618   3.78535533  2.81347752  3.10911345  3.34189892  3.33443713
  2.42448688  2.38472223  3.72806811  3.94998145  3.28642488  2.32396746
  3.64006877  2.6108191   5.04722095  2.24788356  2.98578882  3.16838908
  4.24471569  2.84902191  2.24279356  3.33848739  3.22499871  4.35920501
  4.34456086  3.20160842  2.41525507  3.65808392  1.93648887  4.14033723
  2.80831575  6.26358175  2.83349824  2.60543585  2.68308067  2.74908543
  3.12989736  5.25299883  3.16812944  3.63899302  2.83850837  3.03026295
  3.59622765  2.11758685  3.5503273   2.82376313  3.41406894  3.14427114
  3.20085073  3.88705492  2.93889141  2.84786797  4.08707762  4.48621273
  5.81147885  2.45632052  2.05006194  2.48851633  4.22546959  3.24020219
  2.28753591  3.07918715  4.41707182  3.97422314  5.36070561  3.27088761
  2.75270414  5.79784775  3.23161149  3.33759117  5.17313957  5.83017373
  3.42370343  2.60208249  2.98639297  3.67177391  4.03282499  1.93238616
  4.67125678  3.2497952   2.57149243  4.2254889   3.60026264  4.23713541
  2.57790184  2.14150858  3.81569719  3.16482759  3.68383217  2.51565647
  6.17353797  2.75384927]

average time per mask 3.643783020345788
average time per class
road : 20 masks, 3.2690787434577944 s./mask
sidewalk : 20 masks, 3.4742431640625 s./mask
building : 23 masks, 3.609712611074033 s./mask
wall : 19 masks, 4.265611786591379 s./mask
fence : 20 masks, 4.7114664077758786 s./mask
pole : 22 masks, 3.646247311071916 s./mask
traffic light : 19 masks, 3.6693961243880424 s./mask
traffic sign : 21 masks, 3.3969968046460832 s./mask
vegetation : 18 masks, 3.7958735624949136 s./mask
terrain : 19 masks, 4.09984975112112 s./mask
sky : 19 masks, 3.245059741170783 s./mask
person : 20 masks, 3.190300393104553 s./mask
rider : 18 masks, 4.025683482487996 s./mask
car : 21 masks, 3.1544241337549117 s./mask
truck : 20 masks, 4.225753831863403 s./mask
bus : 20 masks, 3.3451106667518617 s./mask
train : 21 masks, 3.1133357797350203 s./mask
motorcycle : 20 masks, 3.2184825897216798 s./mask
bicycle : 20 masks, 3.9320826172828673 s./mask
12 errors out of 380 annotations, accuracy 96.8 %
"""


def read_image(fname):
    fnima = os.path.join(fname)
    ima = cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        ima = cv2.resize(ima, (1632, 1216))
    return ima


def read_gt(fname):
    gt = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        gt = cv2.resize(gt, (1632, 1216), interpolation=cv2.INTER_NEAREST)
    return gt


def read_masks(fname):
    return np.load(fname, allow_pickle=True)['masks']



def get_ignore_label():
    if args.dataset == 'cityscapes_train':
        return 255
    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        return 19
    return

def make_filenames():
    if args.dataset == 'cityscapes_train':
        filenames = 'cityscapes_train.txt'
        path_dataset = '../data/cityscapes'
        path_images = os.path.join(path_dataset, 'leftImg8bit', 'train')
        path_labels = os.path.join(path_dataset, 'gtFine', 'train')
        path_masks = '../masks_0.86_0.92_400/cityscapes/train'

    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        filenames = 'mapillary_vistas_aspect_1.33_train.txt'
        path_dataset = '../data/mapillary_vistas_aspect_1.33'
        path_images = os.path.join(path_dataset, 'training', 'images')
        path_labels = os.path.join(path_dataset, 'training', 'labels')
        path_masks = '../masks_0.86_0.92_400/mapillary_vistas_aspect_1.33/training'

    with open(filenames, 'r') as f:
        base_fnames = f.read().splitlines()

    base_fnames.sort()
    fnames_images = []
    fnames_groundtruth = []
    fnames_masks = []
    for fn in base_fnames:
        if args.dataset == 'cityscapes_train':
            fn_ima = os.path.join(path_images, fn + '_leftImg8bit.png')
            fn_gt = os.path.join(path_labels, fn + '_gtFine_labelTrainIds.png')
            fn_masks = os.path.join(path_masks, fn + '_masks.npz')

        if args.dataset == 'mapillary_vistas_aspect_1.33_train':
            fn_ima = os.path.join(path_images, fn + '.jpg')
            fn_gt = os.path.join(path_labels, fn + '.png')
            fn_masks = os.path.join(path_masks, fn + '.npz')

        for fn in [fn_ima, fn_gt, fn_masks]:
            assert os.path.isfile(fn), '{} does not exists'.format(fn)

        fnames_images.append(fn_ima)
        fnames_groundtruth.append(fn_gt)
        fnames_masks.append(fn_masks)

    return fnames_images, fnames_groundtruth, fnames_masks



def selected_class(x, y):
    if args.dataset == 'cityscapes_train':
        f = 1
        first_row_legend = 1024
    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        f = 1632 / 2048
        first_row_legend = 1216

    cols_per_class = 205 * f
    rows_per_class = 56 * f
    num_classes_per_row = 10
    c = int(-1 + x // cols_per_class + num_classes_per_row * ((y - first_row_legend) // rows_per_class))
    # assumption: the legend is at the bottom of the image and has its same width
    # -1 + because first class in legend is void
    if y > first_row_legend:
        print(x, y, c, dict_classes[c])
        return c
    else:
        return None


def on_click(event):
    global last_event
    if event.button is MouseButton.LEFT:
        last_event = event


def group_masks_by_label(masks):
    sorted_masks = sorted(masks, key=(lambda x: x['label']))
    grouped_masks = {}
    for k, v in groupby(sorted_masks, key=(lambda x: x['label'])):
        grouped_masks[k] = list(v)
    return grouped_masks


def make_masks_one_image(fn_ima, fn_gt, fn_masks, ignore_label):
    for fn in [fn_ima, fn_gt, fn_masks]:
        assert os.path.exists(fn)

    gt = read_gt(fn_gt)
    masks = read_masks(fn_masks)
    num_masks_ima = len(masks)
    masks_one_image = []
    for nmi in range(num_masks_ima):
        area = masks[nmi]['area']  # or np.sum(seg)
        if area > min_area:
            seg = masks[nmi]['segmentation']
            counts_gt = np.bincount(gt[seg], minlength=args.num_classes)
            majority_label_gt = np.argmax(counts_gt)
            if majority_label_gt != ignore_label:
                # this is to discard the void regions in the groundtruth like the own car motor cover, that is,
                # regions where ignore_label is the most frequent groundtruth label (but not necessarily the only one).
                gt_seg = gt[seg]
                idx = gt_seg != ignore_label
                npixels = idx.sum()
                confusion_matrix = ConfusionMatrix(args.num_classes)
                confusion_matrix.add((majority_label_gt * np.ones(npixels)).astype(dtype=np.int64), gt_seg[idx].astype(np.int64))
                # print('counts_gt', counts_gt)
                # print('unique gt_seg[idx]', np.unique(gt_seg[idx]))
                # print('majority gt label', majority_label_gt)
                # print('cm', confusion_matrix.value())
                # print()
                new_mask = {
                    'fname_ima': fn_ima,
                    'fname_masks': fn_masks,
                    'nmask_in_image': nmi,
                    'label': majority_label_gt,
                    'confusion_matrix': confusion_matrix.value()[:, majority_label_gt],
                    # this is the only non-zero column, because rows = groundtruth, cols = predictions
                    # = the majority label for all pixels in the mask
                    'bbox': masks[nmi]['bbox'],
                }
                masks_one_image.append(new_mask)

    print('.', end='', flush=True)
    return masks_one_image


def make_all_masks(debug=False):
    print('making all_masks parallel...')
    fnames_images, fnames_groundtruth, fnames_masks = make_filenames()
    if debug:
        fnames_images = fnames_images[:100]
        fnames_groundtruth = fnames_groundtruth[:100]
        fnames_masks = fnames_masks[:100]

    ignore_label = get_ignore_label()
    num_images = len(fnames_images)
    num_cores = cpu_count()
    print('{} cores'.format(num_cores))
    with Pool(processes=num_cores) as pool:
        all_masks = pool.starmap(make_masks_one_image,
                                 [(fnames_images[i], fnames_groundtruth[i], fnames_masks[i], ignore_label)
                                  for i in range(num_images)])
    # flatten
    all_masks = [mask for masks_one_image in all_masks for mask in masks_one_image]
    print('done')
    return all_masks


def parse_args():
    parser = argparse.ArgumentParser(description='Computes average time to annotate a mask and the accuracy of '
                                     'the annotations wrt to the majority label of the region pixels, a.k.a '
                                     'region groundtruth. Only for Cityscapes train dataset. This average time serves '
                                     'to translate % budget to number of regions annotated')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num-masks-to-annotate', type=int, default=0,
                        help='Total masks to annotate, being the masks sampled from the pool of all masks, thus '
                             'following its class distribution.')
    parser.add_argument('--num-masks-to-annotate-per-class', type=int, default=0,
                        help='This number of masks is sampled among the masks of each class.')

    args = parser.parse_args()
    assert args.dataset in ['cityscapes_train', 'mapillary_vistas_aspect_1.33_train']
    assert (args.num_masks_to_annotate > 0) ^ (args.num_masks_to_annotate_per_class > 0), \
        'pass either --num-masks-to-annotate or --num-masks-to-annotate-per-class'
    return args


def get_legend():
    legend = plt.imread('cityscapes_legend.png')
    if args.dataset == 'cityscapes_train':
        return legend

    if args.dataset == 'mapillary_vistas_aspect_1.33_train':
        f = 1632 / 2048
        return cv2.resize(legend, (int(f * legend.shape[1]), int(f * legend.shape[0])))


if __name__ == '__main__':
    args = parse_args()

    dict_classes = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
                    6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
                    10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train',
                    17: 'motorcycle', 18: 'bicycle'}
    args.num_classes = len(dict_classes)
    min_area = 1000

    fname_all_masks = 'all_masks_{}.pkl'.format(args.dataset)
    if not os.path.exists(fname_all_masks):
        all_masks = make_all_masks()
        with open(fname_all_masks, 'wb') as f:
            pickle.dump(all_masks, f)
    else:
        with open(fname_all_masks, 'rb') as f:
            all_masks = pickle.load(f)

    if args.num_masks_to_annotate > 0:
        # select masks randomly, thus respecting the prob distribution of the classes
        masks_to_annotate = np.random.choice(all_masks, args.num_masks_to_annotate, replace=False)
    elif args.num_masks_to_annotate_per_class > 0:
        # select same number of masks per class and the shuffle
        masks_to_annotate_per_class = {}
        dict_masks = group_masks_by_label(all_masks)
        for c in range(args.num_classes):
            masks_to_annotate_per_class[c] = np.random.choice(dict_masks[c], args.num_masks_to_annotate_per_class, replace=False)
            print('found {} masks of class {}'.format(len(masks_to_annotate_per_class[c]), dict_classes[c]))

        # put all the masks in dictionary into a single list and shuffle
        masks_to_annotate = np.random.permutation([m for masks_of_a_class in masks_to_annotate_per_class.values()
                                                   for m in masks_of_a_class])
    else:
        assert False

    last_event = None
    region_labels = []
    annotation_labels = []
    annotation_times = []
    legend = get_legend()
    fig = plt.figure()
    for mask in masks_to_annotate:
        fname_ima = mask['fname_ima']
        fname_masks = mask['fname_masks']
        idx_mask = mask['nmask_in_image']
        masks_image = read_masks(mask['fname_masks'])
        seg = masks_image[idx_mask]['segmentation']
        x0, y0, width, height = mask['bbox']
        x1 = x0 + width
        y1 = y0 + height
        cvert = np.argmax(np.sum(seg, axis=0))
        chorz = np.argmax(np.sum(seg, axis=1))
        ima = read_image(fname_ima)
        ima_with_boundary = mark_boundaries(ima, seg, color=(1,1,0), mode='thick')
        ima_to_show = np.vstack([ima_with_boundary, legend])
        region_label = mask['label']
        region_labels.append(region_label)

        plt.imshow(ima_to_show)
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        class_name = dict_classes[region_label]
        plt.text(20, 20, class_name, color='white')
        plt.text(20, 60, class_name, color='black')
        plt.plot([cvert, cvert], [0, y0], 'y:') # 1023
        plt.plot([cvert, cvert], [y1, ima.shape[0]], 'y:')
        plt.plot([0, x0], [chorz, chorz], 'y:') # 2047
        plt.plot([x1, ima.shape[1]], [chorz, chorz], 'y:')
        plt.connect('button_press_event', on_click)
        plt.axis('off')
        plt.show(block=False)
        t1 = time()
        plt.waitforbuttonpress()
        t2 = time()
        x, y = int(last_event.xdata), int(last_event.ydata)
        # coordinates displayed in lower right corner of figure
        annotation_labels.append(selected_class(x, y))
        annotation_times.append(t2-t1)
        plt.clf()
        # print(t2-t1)

    plt.close(fig)

    region_labels = np.array(region_labels)
    annotation_labels = np.array(annotation_labels)
    annotation_times = np.array(annotation_times)

    print('region labels', region_labels)
    print('annotation labels', annotation_labels)
    print('annotation times', annotation_times)
    total_time = sum(annotation_times)
    print('average time per mask {}'.format(total_time/len(masks_to_annotate)))

    print('average time per class')
    num_masks_per_class = np.bincount(annotation_labels, minlength=args.num_classes)
    for c in range(args.num_classes):
        idx = annotation_labels==c
        total_time_this_class = sum(annotation_times[idx])
        num_masks_this_class = sum(idx)
        if num_masks_this_class > 0:
            print('{} : {} masks, {} s./mask'.format(dict_classes[c], num_masks_this_class,
                                                     total_time_this_class / num_masks_this_class))
        else:
            print('{} : no masks'.format(dict_classes[c]))

    num_errors = np.sum(annotation_labels != region_labels)
    print('{} errors out of {} annotations, accuracy {} %'.format(num_errors, len(masks_to_annotate),
          np.round(100 * (1. - num_errors / len(masks_to_annotate)), decimals=1)))

