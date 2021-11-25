import torch.nn.functional as F

from libs.utils import *

class Forward_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

        hough_space = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'hough_space')
        self.hough_h, self.hough_w = hough_space['idx'].shape
        self.hough_space_idx_all = hough_space['idx']
        self.hough_space_idx = to_tensor(hough_space['idx'][self.hough_h // 3:self.hough_h // 3 * 2])

        self.cluster_num = cfg.cluster_num
        self.da = 1
        self.dd = 1

    def run_for_cls(self, img, model):
        out_temp = {}
        out_temp['center_idx'] = torch.LongTensor([]).cuda()
        out_temp['cluster_idx'] = torch.LongTensor([]).cuda()

        model.initialize()
        out = model.forward_for_cls(img, is_training=False)
        score = out['score'].clone()

        for i in range(self.cfg.max_iter):

            # Clustering
            score_temp = score * model.visit_mask
            cluster_mask, cluster, max_score = model.selection_and_removal(score_temp)

            center_idx = cluster['center_idx']
            cluster_idx = cluster['cluster_idx']

            if self.cfg.constrain_max_score == True:
                if i >= 2 and max_score < 0.7:
                    break

            out_temp['center_idx'] = torch.cat((out_temp['center_idx'], center_idx), dim=0)
            out_temp['cluster_idx'] = torch.cat((out_temp['cluster_idx'], cluster_idx), dim=0)
            model.center_mask[:, :, center_idx] = 0
            model.visit_mask[:, :, cluster_idx] = 0

        out_temp['score'] = score
        out_temp['l_feat'] = out['l_feat']
        return out_temp

    def run_for_reg(self, out_cls, model):
        idx = out_cls['center_idx']
        out = model.forward_for_reg(out_cls['l_feat'], idx)
        return out
