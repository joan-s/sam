# faig aquest nova funcio perque els altres scripts son un caos i quan visualitzen les mascares ho han despres d'haver
# aplicat el filtre de area > 1000. A  mes, en el calcul de les regions SAM un altre parametre era que l'area > 400
# pixels. Potser vaig aplicar el nou filtre per no tornar-les a calcular (1.5 o 2 dies).
# Aqui faig per veure-les totes totes, per generar una imatge per la presentacio d'Antonio.

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave


def read_image(path_images, fname, dataset):
    fnima = os.path.join(path_images, fname)
    print(fnima)
    assert os.path.isfile(fnima)
    ima = cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
    if dataset == 'mapillary_vistas_aspect_1.33':
        ima = cv2.resize(ima, (1632, 1216))
    return ima


def read_masks(path_masks, fname):
    fnmasks = os.path.join(path_masks, fname)
    masks = np.load(fnmasks, allow_pickle=True)['masks']
    return masks


def plot_masks(masks):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4)) # np.zeros
    for mask in sorted_masks:
        seg = mask['segmentation']
        random_color = np.concatenate([np.random.random(3), [0.65]]) # 0.35
        img[seg] = random_color
    return img


dataset = 'easyportrait' #'mapillary_vistas_aspect_1.33' 'cityscapes', 'acdc'

if dataset == 'cityscapes':
    path_masks = 'masks_0.86_0.92_400/cityscapes/train'
    path_images = './data/cityscapes/leftImg8bit/train'
    fnames = [
        'bremen/bremen_000083_000019',
        #'weimar/weimar_000017_000019',
        'stuttgart/stuttgart_000157_000019',
        'jena/jena_000075_000019',
        #'bremen/bremen_000067_000019',
        'cologne/cologne_000012_000019',
        'strasbourg/strasbourg_000000_021651',
        #'bremen/bremen_000019_000019',
        #'bochum/bochum_000000_001828',
        #'krefeld/krefeld_000000_034156',
        #'bremen/bremen_000084_000019',
    ]
    fnimas = [fn+'_leftImg8bit.png' for fn in fnames]
    fnmasks = [fn+'_masks.npz' for fn in fnames]

elif dataset == 'mapillary_vistas_aspect_1.33':
    path_masks = './masks_0.86_0.92_400/mapillary_vistas_aspect_1.33/training'
    path_images = './data/mapillary_vistas_aspect_1.33/training/images'
    # fnames = [
    #     '3-ZizFQ1WDvxfgAUkDvcWw',
    #     'z1qtUAgkrLR-3dGGw6MFdg',
    #     'eTlwMXy2gFxMYjDhWV78sQ' # 81 masks
    #     'W4SbbddXg1x2v23VArJLsg' # 99 masks
    #     'XLgr6WNotcTekdHcJgtalw' # 452 masks finestres
    #     'gC_OOs5KJuwixb8hQp823w' # 560 masks finestres i llambordes
    #     'IGFIBamIlRmNq2HlhrXXEQ' # 444 masks
    # ]
    # ratio 4:3
    fnames = [
        #'0035fkbjWljhaftpVM37-g',
        #'014cEvKoAuTJgL6yJvmZow',
        '0155X2oSxdk_osOgaIxUKQ',
        #'016VfPg9PpAc5JHPN3qEVA',
        '01o_TmhYAKongD0M2R4zbQ',
        #'020TJuYqjCSgDVOypUKc7A',
        #'020y16qxpuwigmJH05nTjA',
        #'025hM-2yEUkg_sNVNSCsQQ',
        '028jmhAnOFJO1wYDQW9x7Q',
        #'03U2OOIz3Z6r0VUrl6tBMg',
        #'042yplKtH0jE2nampnv1kg',
        '04m4f91ob7I0XpE36ZDFLw',
        '04MC8ARpOUnkmJsVToTh-A',
        #'04qLNUT3T87a-PBiQjbkxA',
    ]
    fnimas = [fn + '.jpg' for fn in fnames]
    fnmasks = [fn + '.npz' for fn in fnames]

elif dataset == 'acdc':
    path_images = './data/acdc/rgb_anon_trainvaltest/rgb_anon'
    path_masks = './masks_0.86_0.92_400/acdc'
    fnames = [
        'snow/train/GOPR0122/GOPR0122_frame_000161_rgb_anon',
        'rain/train/GOPR0400/GOPR0400_frame_000216_rgb_anon',
        #'rain/train_ref/GOPR0400/GOPR0400_frame_000216_rgb_ref_anon',
        'fog/train/GOPR0475/GOPR0475_frame_000041_rgb_anon',
        'night/train/GOPR0351/GOPR0351_frame_000159_rgb_anon',
    ]
    fnimas = [fn + '.png' for fn in fnames]
    fnmasks = [fn + '.npz' for fn in fnames]
elif dataset == 'easyportrait':
    path_images = '/home/joans/109/datasets/easyportrait/images/train/'
    if True:
        path_masks = './masks_0.86_0.92_400/easyportrait/train'
        fnames = [
            '0a0d730f-38c7-4d11-b65d-cc447038db31',
            '0a1a0089-e855-46d0-94b4-a2485e22f928',
            '0a01bab0-f05c-45a7-aa2e-785feb157e5d',
            '0a4e5b86-50cb-4d6b-a841-67a0cf867481',
            '0a5a6055-76ae-419f-b094-9e63f91a9b6b',
            '0a5f8b1a-f83f-4837-bb58-b095af9bbeb6',
            '0a8e566f-d2c3-4f35-86c2-f9cc2c932b56',
            '0a9d193a-02a7-466d-9e52-f1abf7db9588',
            '0a88db67-825e-4c06-b597-143b7aa2048e',
            '0a91d51e-4184-40e7-b1f8-fd6d038cf85b',
        ]
    if False:
        path_masks = './masks_1e-3_1e-3_50/easyportrait/train'
        fnames = [
            '1ac6d5ad-64f3-45f7-9fb6-915f060ec214',
            '245dd09c-305d-4ba7-a225-d88c970347cf',
            '3138f95d-c7d7-4811-9b11-480864c47af6',
        ]
    if False:
        path_masks = './masks_slic/easyportrait/train'
        fnames = [
            '0f6d654b-27c7-497d-a6de-cb6514cf4284',
            '6e81859c-c5dc-4f7a-91a7-08bc09a31bc4',
            '09e0da01-524f-4bb9-878f-6e0342b6206f',
            '6525760d-44f2-4c47-849d-8401d2c4619f',
            'a3e94401-5841-4bda-a6f7-ffd3436c2a97',
            'a6c0ee7b-5000-4ab0-98d8-609bd6b7b2c8',
        ]
    fnimas = [fn + '.jpg' for fn in fnames]
    fnmasks = [fn + '.npz' for fn in fnames]
else:
    assert False

if __name__ == '__main__':
    for fn, fnima, fnmask in zip(fnames, fnimas, fnmasks):
        ima = read_image(path_images, fnima, dataset)
        masks = read_masks(path_masks, fnmask)
        ima_masks = plot_masks(masks)

        plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(ima)
        plt.axis('off')
        plt.title('{}, {} masks'.format(fnima, len(masks)))
        plt.subplot(2,1,2)
        plt.imshow(ima_masks)
        plt.axis('off')
        plt.show(block=False)

        if False:
            if dataset == 'cityscapes':
                fn = fn.split('/')[1] # remove city directory from name
            imsave(fn+'.png', ima)
            cv2.imwrite(fn + '_masks.png', (255*ima_masks).astype(np.uint8))



