import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils import *
from libs.modules import *
from libs.module_for_region_pooling import *

from networks.backbone import *

class Regression2(nn.Module):
    def __init__(self, size1, size2=256):
        super(Regression2, self).__init__()
        self.linear1 = nn.Linear(size1, size2)
        self.linear2 = nn.Linear(size2, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.sigmoid(out)

        return out

class Regression3(nn.Module):
    def __init__(self, size1, size2=256):
        super(Regression3, self).__init__()
        self.linear1 = nn.Linear(size1, size2)
        self.linear2 = nn.Linear(size2, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out = self.linear1(x)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.sigmoid(out)

        return out

class Regression(nn.Module):
    def __init__(self, channel=256):
        super(Regression, self).__init__()
        self.linear = nn.Linear(channel, 1)

    def forward(self, x):

        x = self.linear(x)
        out = F.sigmoid(x)
        return out

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.size = np.float32(self.cfg.size)
        self.scale_factor = self.cfg.scale_factor
        self.feat_c = 512
        self.sq_c = 256
        self.region_num_list = [2, 3, 4]

        # model
        self.encoder = vgg16(pretrained=True)

        self.regression2 = Regression2(size1=self.feat_c * 4)
        self.regression3 = Regression3(size1=self.feat_c)

        # module
        self.inter_region = Inter_Region(cfg)
        self.adj_region = Adjacent_Region(cfg)


    def global_region_pooling(self, feat_map, mask):
        ep = 1e-7

        b, c, h, w = feat_map.shape

        n = torch.sum(mask != 0, dim=(3, 4), keepdim=True).type(torch.float)
        weight = (n / (torch.sum(n, dim=2, keepdim=True) + ep))

        idx1 = (n == 0).nonzero()[:, 0]
        idx2 = (n == 0).nonzero()[:, 2]

        n[idx1, :, idx2] += 1

        region_feat = torch.sum(mask * feat_map.view(b, c, 1, h, w), dim=(3, 4), keepdim=True) / n
        region_feat = region_feat * weight
        return region_feat[:, :, :, 0, 0].permute(0, 2, 1).contiguous()

    def line_pooling(self, feat_map, line_mask, mode='node'):
        num = torch.sum(line_mask, dim=(2, 3), keepdim=True)
        line_feat = torch.sum(line_mask * feat_map, dim=(2, 3), keepdim=True) / num
        return line_feat[:, :, :, 0].permute(0, 2, 1).contiguous()


    def forward_encoder(self, img, edge_data, is_training=False):

        feat1, feat2 = self.encoder(img)

        self.feat = dict()
        self.feat[self.scale_factor[0]] = feat1
        self.feat[self.scale_factor[1]] = feat2

        if edge_data is not None:
            self.inter_region.update_for_visualize(img, edge_data['edge_idx_set'])
            self.adj_region.update_for_visualize(img, edge_data['edge_idx_set'])

    def forward_node_score(self, data, is_training=False):
        temp_line_feat = torch.FloatTensor([]).cuda()
        temp_node_score = torch.FloatTensor([]).cuda()

        for sf in self.scale_factor:
            # get feat map
            feat_map = self.feat[sf]
            _, c, _, _ = feat_map.shape

            # get preprocessed data
            idx = data['node_idx_set']
            l_mask = data['line_data']['line_mask'][sf].permute(0, 3, 1, 2)
            r_mask = data['line_data']['region_mask'][sf].permute(0, 3, 1, 2)
            b, _, h, w = l_mask.shape
            buffer = torch.zeros((b, 2, self.feat_c), dtype=torch.float).cuda()

            # line pooling
            line_feat = self.line_pooling(feat_map, l_mask, mode='node')
            temp_line_feat = torch.cat((temp_line_feat, line_feat), dim=1)

            # inter region correlation
            inter_feat = self.global_region_pooling(feat_map, r_mask.view(b, 1, 2, h, w))
            inter_feat = torch.cat((inter_feat, buffer), dim=1)

            node_score = self.regression3(line_feat)
            alpha = self.regression2(inter_feat)
            tot_node_score = node_score * alpha
            temp_node_score = torch.cat((temp_node_score, tot_node_score), dim=1)

        tot_node_score = torch.mean(temp_node_score, dim=1, keepdim=True)

        if is_training == True:
            return {'node_score': tot_node_score}
        else:
            return {'node_score': tot_node_score}

    def forward_edge_score(self, data, is_training=False):

        temp_edge_score = torch.FloatTensor([]).cuda()

        for sf in self.scale_factor:
            # get feat map
            feat_map = self.feat[sf]
            _, c, _, _ = feat_map.shape

            # get preprocessed data
            edge_idx = data['edge_idx_set']
            r_mask1 = data['line_data1']['region_mask'][sf].permute(0, 3, 1, 2)
            r_mask2 = data['line_data2']['region_mask'][sf].permute(0, 3, 1, 2)
            l_mask1 = data['line_data1']['line_mask'][sf].permute(0, 3, 1, 2)
            l_mask2 = data['line_data2']['line_mask'][sf].permute(0, 3, 1, 2)

            b, _, h, w = r_mask1.shape

            # line pooling
            line_feat1 = self.line_pooling(feat_map, l_mask1)
            line_feat2 = self.line_pooling(feat_map, l_mask2)

            # inter region correlation
            inter_mask = self.inter_region.get_inter_mask(r_mask1, r_mask2)
            inter_feat = self.global_region_pooling(feat_map, inter_mask.view(b, 1, 4, h, w))

            node_score1 = self.regression3(line_feat1)
            node_score2 = self.regression3(line_feat2)
            alpha = self.regression2(inter_feat)

            tot_node_score = (node_score1 + node_score2) / 2
            edge_score = tot_node_score * alpha

            temp_edge_score = torch.cat((temp_edge_score, edge_score), dim=1)

        tot_edge_score = torch.mean(temp_edge_score, dim=1, keepdim=True)
        if is_training == True:
            return {'edge_score': tot_edge_score}
        else:
            return {'edge_score': tot_edge_score}
