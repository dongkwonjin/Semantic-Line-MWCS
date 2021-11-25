import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *
from libs.modules import *

class Maximum_clique(object):

    def __init__(self, cfg, visualize=None):
        self.cfg = cfg
        self.visualize = visualize
        self.height = cfg.height
        self.width = cfg.width

        self.center = np.array(self.cfg.center_pt)
        self.max_dist = self.cfg.max_dist

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.candidates = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates'))
        self.cand_angle = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'angle'))
        self.cand_dist = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'dist'))

        self.tau = 0.5
        self.MIN_VAL = -1000

        self.encode_bitlist()
        self.decode_bitlist()
        self.encode_edge_idx()
        self.match_bitlist_with_edge()

    def encode_bitlist(self):
        self.num_node = self.cfg.max_iter
        num_clique = 2**self.num_node

        bitlist = torch.LongTensor([]).cuda()
        for i in range(1, num_clique):

            k = i
            bit = torch.zeros((1, self.num_node), dtype=torch.int64).cuda()
            for j in range(self.num_node):
                rest = k % 2
                k //= 2
                bit[0, j] = rest
                if k == 0:
                    break
            if torch.sum(bit) == 1:  # case of node
                continue
            bitlist = torch.cat((bitlist, bit))

        self.bitlist = bitlist
        self.num_clique = self.bitlist.shape[0]

    def decode_bitlist(self):

        check = torch.zeros((2**self.num_node), dtype=torch.int64).cuda()
        for i in range(self.num_clique):

            bit = self.bitlist[i]
            k = 1
            m = 0
            for j in range(self.num_node):
                m += (k * bit[j])
                k *= 2
            check[m] = 1

        idx = (check == 0).nonzero()
        if idx.shape[0] == 0:
            print('Successfully encoded')
        else:
            print("error code : {}".format(idx))

    def encode_edge_idx(self):

        edge_idx = torch.zeros((self.num_node, self.num_node), dtype=torch.int64).cuda()
        k = 0
        for i in range(self.num_node):
            for j in range(i+1, self.num_node):
                edge_idx[i, j] = k
                k += 1

        self.edge_idx = edge_idx
        self.edge_max_num = torch.max(edge_idx)

    def match_bitlist_with_edge(self):

        clique_idxlist = torch.LongTensor([]).cuda()
        for i in range(self.num_clique):

            bit = self.bitlist[i]
            nodelist = bit.nonzero()
            num_node = nodelist.shape[0]
            idx_check = torch.zeros((1, self.edge_max_num + 1), dtype=torch.int64).cuda()
            for j in range(num_node):
                for k in range(j+1, num_node):
                    idx_check[0, self.edge_idx[nodelist[j, 0], nodelist[k, 0]]] = 1

            clique_idxlist = torch.cat((clique_idxlist, idx_check))

        self.clique_idxlist = clique_idxlist
        self.clique_idxnum = torch.sum(clique_idxlist, dim=1)

    def run(self):

        clique_energy = torch.sum((self.edge_score * self.clique_idxlist), dim=1)

        idx_max = torch.argmax(clique_energy)
        if clique_energy[idx_max] > self.tau:
            mul_idx = self.bitlist[idx_max].nonzero()[:, 0]
            pri_idx = torch.argmax(self.node_score[mul_idx]).view(1)
        else:
            mul_idx = (self.node_idx == self.pri_node_idx).nonzero()[:, 0]
            pri_idx = torch.argmax(self.node_score[mul_idx]).view(1)

        return {'cls_pri_pts': self.node_pts[mul_idx[pri_idx]],
                'cls_mul_pts': self.node_pts[mul_idx],
                'cls_idx': self.node_idx[mul_idx],
                'cls_pri_idx': pri_idx}

    def update(self, batch, out, mode, dataset_name, pair_data=None):
        self.mode = mode
        self.dataset_name = dataset_name

        self.img_name = batch['img_name'][0]

        self.node_score = out['node_score'][:, 0]
        self.edge_score = out['edge_score'][:, 0]
        idx = (self.edge_score <= self.tau)
        self.edge_score[idx] = self.MIN_VAL
        self.edge_score = self.edge_score.view(1, -1)

        self.node_idx = out['center_idx']
        self.node_pts = self.candidates[self.node_idx]
        self.pri_node_idx = self.node_idx[torch.argmax(self.node_score)]


    def run_for_reg(self, out, out_cls):

        # out hough line
        idx = out_cls['cls_idx']
        cls_angle = self.cand_angle[idx]
        cls_dist = self.cand_dist[idx]

        angle_offset = out['offset'][0, :, 0]
        dist_offset = out['offset'][0, :, 1]

        reg_angle = (cls_angle + angle_offset * self.cfg.max_offset_a)
        reg_angle2 = reg_angle.clone()
        reg_angle2[reg_angle < -90] += 180
        reg_angle2[reg_angle > 90] -= 180

        cls_dist[reg_angle < -90] *= -1
        cls_dist[reg_angle > 90] *= -1
        dist_offset[reg_angle < -90] = 0
        dist_offset[reg_angle > 90] = 0
        reg_dist2 = (cls_dist + dist_offset * self.cfg.max_offset_d)

        line_pts = []
        for i in range(reg_angle2.shape[0]):
            out_pts = self.convert_to_line(reg_angle2[i], reg_dist2[i])
            if out_pts.shape[0] == 4:
                line_pts.append(out_pts)
        mul_pts = np.float32(line_pts)
        pri_pts = mul_pts[out_cls['cls_pri_idx']].reshape(1, -1)

        return {'reg_pri_pts': pri_pts,
                'reg_mul_pts': mul_pts}

    def convert_to_line(self, angle, dist):
        angle = to_np2(angle)
        dist = to_np2(dist)
        a = np.tan(angle / 180 * math.pi)

        if angle != -90:
            b = -1
            c = dist * np.sqrt(a ** 2 + b ** 2) - (a * self.center[0] + b * self.center[1])
        else:
            a = 1
            b = 0
            c = dist + self.center[0]

        line_pts = find_endpoints_from_line_eq(line_eq=[a, b, c], size=[self.width - 1, self.height - 1])
        return line_pts
