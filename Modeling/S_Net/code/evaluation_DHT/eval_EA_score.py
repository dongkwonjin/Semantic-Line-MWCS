from libs.utils import *

import numpy as np
import os
from evaluation_DHT.utils import caculate_precision, caculate_recall

def eval_EA_score_pickle(cfg, pred_path, gt_path, dataset_name, logfile):
    datalist = load_pickle(pred_path + 'test_' + dataset_name + '/pickle/datalist')

    total_precision = np.zeros(99)
    total_recall = np.zeros(99)
    nums_precision = 0
    nums_recall = 0
    for i in range(len(datalist)):
        filename = datalist[i]
        data = load_pickle(pred_path + 'test_' + dataset_name + '/pickle/' + filename[:-4])

        pred = to_np(data['out']['mul'])[:, [1, 0, 3, 2]]
        gt = np.int32((to_np(data['gt']['mul'])[:, [1, 0, 3, 2]] + 1)).tolist()


        for i in range(1, 100):
            p, num_p = caculate_precision(pred.tolist(), gt, thresh=i*0.01)
            r, num_r = caculate_recall(pred.tolist(), gt, thresh=i*0.01)
            total_precision[i-1] += p
            total_recall[i-1] += r
        nums_precision += num_p
        nums_recall += num_r

    total_recall = total_recall / nums_recall
    total_precision = total_precision / nums_precision
    f = 2 * total_recall * total_precision / (total_recall + total_precision)

    print('Mean P:', total_precision.mean())
    print('Mean R:', total_recall.mean())
    print('Mean F:', f.mean())
    logger('Mean P: %5f\n'
           'Mean R: %5f\n'
           'Mean F: %5f\n' % (total_precision.mean(), total_recall.mean(), f.mean()), logfile)


def eval_EA_score_npy(cfg, pred_path, gt_path, dataset_name, logfile):
    pred_path = pred_path + 'visualize_' + dataset_name + '/visualize_test/'
    filenames = sorted(os.listdir(pred_path))

    total_precision = np.zeros(99)
    total_recall = np.zeros(99)
    nums_precision = 0
    nums_recall = 0
    k = 0

    if dataset_name == 'SEL_Hard':
        data = load_pickle(cfg.dir['dataset']['SEL']['SEL_Hard'] + 'data/test')

    for filename in filenames:

        if 'npy' not in filename:
            continue
        pred = np.load(os.path.join(pred_path, filename))

        if dataset_name == 'SEL_Hard':
            gt = np.int32((data['multiple'][k][:, [1, 0, 3, 2]] + 1)).tolist()
        else:
            gt_txt = open(os.path.join(gt_path, filename.split('.')[0] + '.txt'))
            gt_coords = gt_txt.readlines()
            gt = [[int(float(l.rstrip().split(', ')[1])), int(float(l.rstrip().split(', ')[0])), int(float(l.rstrip().split(', ')[3])), int(float(l.rstrip().split(', ')[2]))] for l in gt_coords]

        for i in range(1, 100):
            p, num_p = caculate_precision(pred.tolist(), gt, thresh=i*0.01)
            r, num_r = caculate_recall(pred.tolist(), gt, thresh=i*0.01)
            total_precision[i-1] += p
            total_recall[i-1] += r
        nums_precision += num_p
        nums_recall += num_r

        k += 1

    total_recall = total_recall / nums_recall
    total_precision = total_precision / nums_precision
    f = 2 * total_recall * total_precision / (total_recall + total_precision)

    print('Mean P:', total_precision.mean())
    print('Mean R:', total_recall.mean())
    print('Mean F:', f.mean())
    logger('Mean P: %5f\n'
           'Mean R: %5f\n'
           'Mean F: %5f\n' % (total_precision.mean(), total_recall.mean(), f.mean()), logfile)
