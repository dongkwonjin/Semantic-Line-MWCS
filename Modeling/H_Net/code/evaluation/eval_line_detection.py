import torch
from sklearn.metrics import auc

from libs.utils import *

from evaluation_DHT.utils import caculate_precision, caculate_recall

class Evaluation_Semantic_Line(object):

    def __init__(self, cfg, eval_func, eval_hiou):
        self.cfg = cfg
        self.eval_func = eval_func
        self.eval_hiou = eval_hiou

    def measure_AUC_PRF(self, datalist, mode='test', dataset_name='SEL', with_reg=True):
        num = len(datalist)
        miou, match = create_eval_dict()

        for i in range(num):
            data = load_pickle(self.cfg.dir['out'] + mode + '_' + dataset_name + '/pickle/' + datalist[i][:-4])
            if with_reg == False:
                out = data['out']['mul_cls']
            elif with_reg == True:
                out = data['out']['mul_reg']
            gt = data['gt']['mul'].cuda()
            out_num = out.shape[0]
            gt_num = gt.shape[0]

            if gt_num == 0:
                match['r'][i] = torch.zeros(gt_num, dtype=torch.float32).cuda()
            elif out_num == 0:
                match['p'][i] = torch.zeros(out_num, dtype=torch.float32).cuda()
            else:
                miou['p'][i], miou['r'][i] = self.eval_func.measure_miou(out, gt)
                match['p'][i], match['r'][i] = self.eval_func.matching(miou, i)

        # performance
        auc_p, thresds, precision = self.eval_func.calculate_AUC(miou=match, metric='precision')
        auc_r, _, recall = self.eval_func.calculate_AUC(miou=match, metric='recall')
        try:
            f = 2 * precision * recall / (precision + recall)  # F1-score
        except:
            f = 0
        f[np.isnan(f).nonzero()[0]] = 0
        auc_f = auc(thresds[10:191], f[10:191]) / 0.9

        print('F1-score : %f' % (auc_f))
        print('%s ==> AUC_P %5f / AUC_R %5f / AUC_F %5f' % (dataset_name, auc_p, auc_r, auc_f))
        print('%s ==> AUC_P %5f / AUC_R %5f' % (dataset_name, auc_p, auc_r))

        return auc_p, auc_r, auc_f


    def measure_AUC_A(self, datalist, mode='test', dataset_name='SEL', with_reg=True):
        num = len(datalist)
        miou, match = create_eval_dict()

        for i in range(num):
            data = load_pickle(self.cfg.dir['out'] + mode + '_' + dataset_name + '/pickle/' + datalist[i][:-4])
            if with_reg == False:
                out = data['out']['pri_cls']
            elif with_reg == True:
                out = data['out']['pri_reg']
            gt = data['gt']['pri'].cuda()

            miou['a'][i], _ = self.eval_func.measure_miou(out, gt)

        # performance
        auc_a, _, _ = self.eval_func.calculate_AUC(miou=miou, metric='accuracy')

        print('%s ==> AUC_A %5f' % (dataset_name, auc_a))

        return auc_a

    def measure_EA_score(self, datalist, mode='test', dataset_name='SEL', with_reg=True):
        num = len(datalist)

        total_precision = np.zeros(99)
        total_recall = np.zeros(99)
        nums_precision = 0
        nums_recall = 0

        for i in range(num):
            data = load_pickle(self.cfg.dir['out'] + mode + '_' + dataset_name + '/pickle/' + datalist[i][:-4])

            if with_reg == False:
                pred = data['out']['mul_cls'][:, [1, 0, 3, 2]]
            elif with_reg == True:
                pred = data['out']['mul_reg'][:, [1, 0, 3, 2]]
            pred = to_np(pred)
            gt = np.int32(to_np(data['gt']['mul'])[:, [1, 0, 3, 2]]).tolist()

            for i in range(1, 100):
                p, num_p = caculate_precision(pred.tolist(), gt, thresh=i * 0.01)
                r, num_r = caculate_recall(pred.tolist(), gt, thresh=i * 0.01)
                total_precision[i - 1] += p
                total_recall[i - 1] += r
            nums_precision += num_p
            nums_recall += num_r

        total_recall = total_recall / nums_recall
        total_precision = total_precision / nums_precision
        f = 2 * total_recall * total_precision / (total_recall + total_precision)
        f[np.isnan(f).nonzero()[0]] = 0
        print('Mean P:', total_precision.mean())
        print('Mean R:', total_recall.mean())
        print('Mean F:', f.mean())

        mean_p = total_precision.mean()
        mean_r = total_recall.mean()
        mean_f = f.mean()

        return mean_p, mean_r, mean_f

    def measure_HIoU(self, datalist, mode='test', dataset_name='SEL', with_reg=True):
        num = len(datalist)

        IOU_total, F_total = 0, 0
        N = 0
        self.eval_hiou.update_dataset_name(dataset_name)
        for i in range(num):
            data = load_pickle(self.cfg.dir['out'] + mode + '_' + dataset_name + '/pickle/' + datalist[i][:-4])
            if with_reg == False:
                out = data['out']['mul_cls']
            elif with_reg == True:
                out = data['out']['mul_reg']
            gt = data['gt']['mul'].cuda()

            result = self.eval_hiou.run(out, gt, datalist[i])
            if type(result['IOU']) == str:
                continue
            else:
                IOU_total += result['IOU']
                N += 1

        IOU_avg = IOU_total / N
        print('Avg HIoU : {}'.format(IOU_avg))
        return IOU_avg

def create_eval_dict():
    # a : accuracy / p : precision / r : recall

    miou = {'a': {},
            'p': {},
            'r': {}}

    match = {'p': {},
             'r': {}}

    return miou, match
