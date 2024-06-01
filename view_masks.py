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

    img = np.zeros((sorted_masks[0]['segmentation'].shape[0], sorted_masks[0]['segmentation'].shape[1], 4))
    for mask in sorted_masks:
        seg = mask['segmentation']
        random_color = np.concatenate([np.random.random(3), [0.65]]) # 0.35
        img[seg] = random_color
    return img


dataset = 'cityscapes' #'mapillary_vistas_aspect_1.33'
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
else:
    assert False

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

    if dataset == 'cityscapes':
        fn = fn.split('/')[1] # remove city directory from name
    imsave(fn+'.png', ima)
    cv2.imwrite(fn + '_masks.png', (255*ima_masks).astype(np.uint8))



