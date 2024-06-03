import pickle
import numpy as np
import matplotlib.pyplot as plt


def print_classification_report(conf_mat, names):
    tp = np.diag(conf_mat)
    fp = conf_mat.sum(axis=0) - tp
    fn = conf_mat.sum(axis=1) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = tp.sum() / conf_mat.sum()
    support = conf_mat.sum(axis=1).astype(int)
    num_classes = len(conf_mat)
    for c in range(num_classes):
        print('{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{}'.format(c, names[c], precision[c], recall[c], f1[c], support[c]))
    print('accuracy {:.2f}'.format(accuracy))
    print('average precision {:.2f}, recall {:.2f} , f1 {:.2f}'.format(precision.mean(), recall.mean(), f1.mean()))



dict_classes = {0: 'road', 1: 'sidewalk', 2: 'building', 3: 'wall', 4: 'fence', 5: 'pole',
                6: 'traffic light', 7: 'traffic sign', 8: 'vegetation', 9: 'terrain',
                10: 'sky', 11: 'person', 12: 'rider', 13: 'car', 14: 'truck', 15: 'bus', 16: 'train',
                17: 'motorcycle', 18: 'bicycle'}
num_classes = 19

for fname_all_masks in ['all_masks_cityscapes_train.pkl', 'all_masks_mapillary_vistas_aspect_1.33_train.pkl']:
    with open(fname_all_masks, 'rb') as f:
        all_masks = pickle.load(f)

    cm = np.zeros((num_classes, num_classes))
    for mask in all_masks:
        label = mask['label']
        col_cm = mask['confusion_matrix']
        cm[:, label] += col_cm

    cm_normalized = cm / cm.sum(1).clip(min=1e-12)[:, None]

    plt.figure(figsize=(10,8))
    plt.imshow(cm_normalized)
    plt.xlabel('Ground truth',fontsize=16)
    plt.ylabel('Prediction', fontsize=16)
    #plt.title('normalized confusion matrix')
    plt.yticks(range(num_classes), [dict_classes[i] for i in range(num_classes)])
    plt.xticks(range(num_classes), [dict_classes[i] for i in range(num_classes)], rotation=90)
    plt.colorbar()
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.96)
    plt.show(block=False)

    print_classification_report(cm, [dict_classes[c] for c in range(num_classes)])


"""
Cityscapes

0       road            0.97    1.00    0.98    2329519556
1       sidewalk        0.97    0.83    0.89    392897596
2       building        0.98    0.99    0.99    2216369574
3       wall            0.95    0.93    0.94    56691243
4       fence           0.94    0.88    0.91    68911564
5       pole            0.89    0.75    0.81    68283970
6       traffic light   0.90    0.79    0.84    16749506
7       traffic sign    0.91    0.87    0.89    52260630
8       vegetation      0.98    0.97    0.98    1321209982
9       terrain         0.95    0.93    0.94    71313428
10      sky             0.98    0.99    0.98    289717563
11      person          0.99    0.98    0.98    106641348
12      rider           0.96    0.96    0.96    12066978
13      car             1.00    0.99    0.99    673072719
14      truck           0.99    0.99    0.99    27933516
15      bus             0.99    1.00    0.99    25915230
16      train           0.99    0.99    0.99    27527178
17      motorcycle      0.96    0.95    0.96    8161237
18      bicycle         0.97    0.95    0.96    30631021

accuracy 0.98
average precision 0.96, recall 0.93 , f1 0.95

Mapillary

0       road            0.98    0.99    0.99    6817119360
1       sidewalk        0.96    0.91    0.93    1220457192
2       building        0.97    0.99    0.98    6098631165
3       wall            0.95    0.95    0.95    362375931
4       fence           0.95    0.91    0.93    519574127
5       pole            0.91    0.83    0.87    316045866
6       traffic light   0.88    0.88    0.88    89212679
7       traffic sign    0.92    0.94    0.93    227705839
8       vegetation      0.98    0.97    0.97    6024271758
9       terrain         0.95    0.95    0.95    366719471
10      sky             0.99    1.00    1.00    9594149025
11      person          0.98    0.98    0.98    143959025
12      rider           0.97    0.97    0.97    29277589
13      car             1.00    0.99    0.99    1743237677
14      truck           0.99    0.99    0.99    204017982
15      bus             0.99    0.99    0.99    150407242
16      train           0.98    0.99    0.98    13892923
17      motorcycle      0.97    0.97    0.97    26560031
18      bicycle         0.97    0.96    0.97    29283735

accuracy 0.98
average precision 0.96, recall 0.96 , f1 0.96

"""