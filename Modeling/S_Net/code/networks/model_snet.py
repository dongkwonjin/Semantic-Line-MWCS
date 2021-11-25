import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.backbone import *
from libs.utils import *
from libs.modules import *

from libs.module_for_region_pooling import *

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.max_dist = self.cfg.max_dist
        self.sf = cfg.scale_factor
        # candidates
        self.size = np.float32(self.cfg.size)
        self.sf = self.cfg.scale_factor
        self.seg_size = np.int32(self.size / self.sf[1])

        # load candidate lines
        candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.c_num = candidates.shape[0]
        c_idx = torch.arange(self.c_num)
        # generate line mask
        sep_region = Separated_Region(cfg)
        line_data = sep_region.run(c_idx)
        self.line_mask = line_data['line_mask']
        self.line_n = dict()
        for sf in self.sf:
            n, h, w, _ = self.line_mask[sf].shape
            self.line_mask[sf] = self.line_mask[sf].view(1, 1, n, h, w).type(torch.float)
            self.line_n[sf] = torch.sum(self.line_mask[sf], dim=[3, 4])

        angle = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'angle')
        dist = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'dist')
        hough_space = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'hough_space')

        self.cluster_num = self.cfg.cluster_num
        self.angle = to_tensor(angle).view(1, 1, self.c_num) / 90
        self.dist = to_tensor(dist).view(1, 1, self.c_num) / self.max_dist
        self.center = np.array(self.cfg.center_pt)
        self.hough_space = hough_space
        self.hough_h, self.hough_w = self.hough_space['idx'].shape
        self.hough_space_idx = to_tensor(self.hough_space['idx'][self.hough_h // 3:self.hough_h // 3 * 2]).type(torch.long)
        self.inlier_mask = (self.hough_space_idx != -1).view(1, 1, self.hough_h // 3, self.hough_w)
        self.inlier_mask2 = (self.hough_space_idx != -1).view(1, 1, self.hough_h // 3, self.hough_w).type(torch.float)
        self.hough_cand_idx = self.inlier_mask[0, 0].nonzero()
        self.angle_list = to_tensor(np.float32(self.hough_space['angle_list'])) / 90
        self.dist_list = to_tensor(np.float32(self.hough_space['dist_list'])) / self.max_dist

        self.ca = 2
        self.cd = 2
        self.ua = 1
        self.ud = 1
        self.pa = 1
        self.pd = 1

        self.c_sq = 1

        # model
        self.encoder = vgg16(pretrained=True)

        # filter
        self.feat_squeeze = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False))

        self.squeeze = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, self.c_sq, kernel_size=1, bias=False),
            nn.Sigmoid())

        self.squeeze2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, kernel_size=1, bias=False))

    def extract_line_feat(self, feat_map, idx):
        b, c, h, w = feat_map.shape
        line_feat = torch.sum(self.line_mask[idx] * feat_map.view(b, c, 1, h, w), dim=(3, 4)) / self.line_n[idx]

        return line_feat

    def selection_and_removal(self, score):
        idx = torch.sort(score, descending=True)[1][:, 0]

        a_idx, d_idx = (self.hough_space_idx == idx[:, 0]).nonzero()[0]

        cluster = {}
        cluster['angle'] = self.angle_list[a_idx]
        cluster['dist'] = self.dist_list[d_idx]

        a_idx += self.hough_h // 3

        h, w = self.hough_space['idx'].shape
        check = torch.zeros((h, w), dtype=torch.int64)
        x1 = np.maximum(to_np(a_idx) - self.ca, 0)
        x2 = np.minimum(to_np(a_idx) + self.ca + 1, h)
        y1 = np.maximum(to_np(d_idx) - self.cd, 0)
        y2 = np.minimum(to_np(d_idx) + self.cd + 1, w)

        region_idx = to_tensor(self.hough_space['idx'][x1:x2, y1:y2]).type(torch.long).reshape(-1)

        check[x1:x2, y1:y2] = 1
        region_a_idx = check.nonzero()[:, 0]
        region_d_idx = check.nonzero()[:, 1]

        mask = torch.zeros((1, 1, self.c_num)).cuda()
        mask[:, :, region_idx] = 1

        cluster['a_idx'] = region_a_idx
        cluster['d_idx'] = region_d_idx
        cluster['ca_idx'] = a_idx
        cluster['cd_idx'] = d_idx
        cluster['center_idx'] = idx[:, 0]
        cluster['cluster_idx'] = region_idx

        return mask, cluster, score[0, 0, idx[:, 0]]

    def initialize(self):
        self.center_mask = torch.ones((1, 1, self.c_num)).cuda()
        self.visit_mask = torch.ones((1, 1, self.c_num)).cuda()

    def forward_for_cls(self, img, is_training=True):

        # Feature extraction
        feat1, feat2 = self.encoder(img)
        sq_feat1 = self.feat_squeeze(feat1)
        sq_feat2 = self.feat_squeeze(feat2)

        # Line feature extraction
        l_feat1 = self.extract_line_feat(sq_feat1, self.sf[0])
        l_feat2 = self.extract_line_feat(sq_feat2, self.sf[1])

        l_feat = torch.cat((l_feat1, l_feat2), dim=1)  # [b, c', n] -> [b, 2c', n]
        score = self.squeeze(l_feat)  # [b, 2c', n] -> [b, 1, n]

        return {'score': score,
                'l_feat': l_feat}


    def forward_for_reg(self, l_feat, idx):
        offset = self.squeeze2(l_feat[:, :, idx])  # [b, 2c', n] -> [b, 1, n]

        return {'offset': offset.permute(0, 2, 1)}

