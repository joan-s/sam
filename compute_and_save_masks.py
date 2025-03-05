import os
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import argparse
import numpy as np
from skimage.io import imread

"""
python -i compute_and_save_masks.py \
  --gpu 0 \
  --filename cityscapes_train.txt \
  --dataset cityscapes \
  --vit-model vit_h \
  --path-out cityscapes2/train

python -i compute_and_save_masks.py \
  --gpu 0 \
  --filename mapillary_14716_train_images_aspect_1.33_1.txt \
  --dataset mapillary_vistas_aspect_1.33 \
  --vit-model vit_h \
  --path-out masks_0.86_0.92_400/Mapillary_Vistas_aspect_1.33/training

python -i compute_and_save_masks.py \
  --gpu 1 \
  --filename mapillary_14716_train_images_aspect_1.33_2.txt \
  --dataset mapillary_vistas_aspect_1.33 \
  --vit-model vit_h \
  --path-out masks_0.86_0.92_400/Mapillary_Vistas_aspect_1.33/training  

ls ./data/acdc/rgb_anon_trainvaltest/rgb_anon/fog/*/*/*.png -c1 > acdc_fog_train_val_test.txt
ls ./data/acdc/rgb_anon_trainvaltest/rgb_anon/night/*/*/*.png -c1 > acdc_night_train_val_test.txt
ls ./data/acdc/rgb_anon_trainvaltest/rgb_anon/rain/*/*/*.png -c1 > acdc_rain_train_val_test.txt
ls ./data/acdc/rgb_anon_trainvaltest/rgb_anon/snow/*/*/*.png -c1 > acdc_snow_train_val_test.txt
  
ls ./data/acdc/rgb_anon_trainvaltest/rgb_anon/*/*/*/*.png -c1 > acdc_train_val_test.txt

python -i compute_and_save_masks.py \
  --gpu 0 \
  --filename acdc_train_val_test1.txt \
  --dataset acdc \
  --vit-model vit_h \
  --path-out masks_0.86_0.92_400/acdc
  
  
python -i compute_and_save_masks.py \
  --gpu 0 \
  --filename easyportrait_train_1.txt \
  --dataset easyportrait \
  --downsample 2 \
  --vit-model vit_b \
  --pred-iou-thresh 1e-3 \
  --stability-score-thresh 1e-3 \
  --min-mask-region-area 50 \
  --path-out masks_1e-3_1e-3_50/easyportrait  

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Computes and saves SAM masks for all the images specified in a file.')
    parser.add_argument('--gpu', required=True,
                        help='GPU number, starting at 0')
    parser.add_argument('--filename', required=True,
                        help='text file with the names of the images to be processed')
    parser.add_argument('--dataset', required=True,
                        help='either cityscapes of mapillary, this is to decide the format of the masks filenames '
                        'and if to resize the image')
    parser.add_argument('--downsample', type=int, required=False, default=1,
                        help='downsampling step of input images.')
    parser.add_argument('--vit-model', required=True,
                        help='either vit_h, vit_l, vit_b from larger to smaller in nummber of parameters.')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.86,
                        help='https://github.com/facebookresearch/segment-anything')
    parser.add_argument('--stability-score-thresh', type=float, default=0.92,
                        help='https://github.com/facebookresearch/segment-anything')
    parser.add_argument('--min-mask-region-area', type=int, default=400,
                        help='https://github.com/facebookresearch/segment-anything')
    parser.add_argument('--path-out', required=True,
                        help='directory where the binary map for the SAM regions of each image will be saved, '
                             'one file with all the binary maps per image')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    assert os.path.exists(args.filename)
    assert args.dataset in ['cityscapes', 'mapillary_vistas_aspect_1.33', 'acdc', 'easyportrait']
    assert args.vit_model in ['vit_h', 'vit_l', 'vit_b']
    # because we will run this in parallel, one process per gpu
    # assert not os.path.isdir(args.path_out)
    if not os.path.isdir(args.path_out):
        os.makedirs(args.path_out)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.vit_model == 'vit_h':
        checkpoint = './sam_vit_h_4b8939.pth'
    elif args.vit_model == 'vit_l':
        checkpoint = './sam_vit_l_0b3195.pth'
    elif args.vit_model == 'vit_b':
        checkpoint = './sam_vit_b_01ec64.pth'
    else:
        assert False, 'invalid model {}'.format(args.vit_model)

    sam = sam_model_registry[args.vit_model](checkpoint=checkpoint)
    sam.to(device="cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=2,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=args.min_mask_region_area,  # Requires open-cv to run post-processing
    )
    # Cityscapes 1024 x 2048
    # 0.86, 0.92, 400 -> 1er frame 305 regions, 40 s. => 33 h
    # 0.50, 0.50, 100 -> 1er frame 681 regions, 54 s. => 45 h

    with open(args.filename, 'r') as f:
        fnames = f.read().splitlines()

    for fn in tqdm(fnames):
        if args.dataset == 'easyportrait':
            fn = '/home/joans/109/datasets/easyportrait/images/train/' + fn

        image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        if args.downsample > 1:
            image = image[::args.downsample, ::args.downsample, :]

        if args.dataset == 'cityscapes':
            fname_mask = fn.split('/')[-1].replace('leftImg8bit', 'masks').replace('.png', '.npz')
            # to be read like
            # masks = np.load(fname_ima.replace('leftImg8bit', 'masks').replace('.png', '_masks.npz'),
            #                 allow_pickle=True)['masks_0.86_0.92_400']
        elif args.dataset == 'mapillary':
            image = cv2.resize(image, (1632, 1216))
            fname_mask = fn.split('/')[-1].replace('.jpg', '.npz')
        elif args.dataset == 'acdc':
            image = cv2.resize(image, (1920, 1080))
            condition, split, sequence, fname_image = fn.split('/')[-4:]
            fname_mask = fname_image.replace('.png', '.npz')
        elif args.dataset == 'easyportrait':
            _, _, split, fname_image = fn.split('/')[-4:]
            fname_mask = fname_image.replace('.jpg', '.npz')
        else:
            assert False

        masks = mask_generator.generate(image)
        # print(fn, len(masks))
        if args.dataset in ['cityscapes', 'mapillary']:
            fname_out = os.path.join(args.path_out, fname_mask)
        elif args.dataset == 'acdc':
            dir_output_name = os.path.join(args.path_out, condition, split, sequence)
            if not os.path.isdir(dir_output_name):
              os.makedirs(dir_output_name)
              print('made output dir ' + dir_output_name)

            fname_out = os.path.join(dir_output_name, fname_mask)
        elif args.dataset == 'easyportrait':
            dir_output_name = os.path.join(args.path_out, split)
            if not os.path.isdir(dir_output_name):
              os.makedirs(dir_output_name)
              print('made output dir ' + dir_output_name)

            fname_out = os.path.join(dir_output_name, fname_mask)
        else:
            assert False

        assert not os.path.isfile(fname_out), fname_out + ' already exists'
        np.savez_compressed(fname_out, masks=masks)
        # break




            
