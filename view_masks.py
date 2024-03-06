# faig aquest nova funcio perque els altres scripts son un caos i quan visualitzen les mascares ho han despres d'haver
# aplicat el filtre de area > 1000. A  mes, en el calcul de les regions SAM un altre parametre era que l'area > 400
# pixels. Potser vaig aplicar el nou filtre per no tornar-les a calcular (1.5 o 2 dies).
# Aqui faig per veure-les totes totes, per generar una imatge per la presentacio d'Antonio.

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def read_image(path_images, fname):
    fnima = os.path.join(path_images, fname)
    ima = cv2.cvtColor(cv2.imread(fnima), cv2.COLOR_BGR2RGB)
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


dataset = 'mapillary'
if dataset == 'cityscapes':
    path_masks = 'masks_0.86_0.92_400/cityscapes/train'
    path_images = './data/cityscapes/leftImg8bit/train'
    fnima = 'bremen/bremen_000083_000019_leftImg8bit.png'
    fnmasks = 'bremen/bremen_000083_000019_masks.npz'
elif dataset == 'mapillary':
    path_masks = './mapillary'
    path_images = './data/Mapillary_Vistas_aspect_1.33/training/images'
    fn = '3-ZizFQ1WDvxfgAUkDvcWw'
    # 'z1qtUAgkrLR-3dGGw6MFdg' # 77 masks
    # 'eTlwMXy2gFxMYjDhWV78sQ' # 81 masks
    # 'W4SbbddXg1x2v23VArJLsg' # 99 masks
    #'XLgr6WNotcTekdHcJgtalw' # 452 masks finestres
    # 'gC_OOs5KJuwixb8hQp823w' # 560 masks finestres i llambordes
    # 'IGFIBamIlRmNq2HlhrXXEQ' # 444 masks
    # '3-ZizFQ1WDvxfgAUkDvcWw' 'z1uGUfJJePu5QSFIBuXZGQ' '028jmhAnOFJO1wYDQW9x7Q' '014cEvKoAuTJgL6yJvmZow' '0035fkbjWljhaftpVM37-g'
    fnima = fn + '.jpg'
    fnmasks = fn + '.npz'
else:
    assert False

ima = read_image(path_images, fnima)
masks = read_masks(path_masks, fnmasks)
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

