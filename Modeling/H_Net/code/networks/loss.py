import numpy as np

import torch
import torch.nn as nn

from libs.utils import *

class Loss_Function_MSE(nn.Module):
    def __init__(self):
        super(Loss_Function_MSE, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.weight = 1

    def forward(self, out, train_data1, train_data2):
        node_pos_loss, node_neg_loss = self.balanced_MSE_loss(out['node_score'], train_data1['gt_node_score'], tau=0)
        edge_pos_loss, edge_neg_loss = self.balanced_MSE_loss(out['edge_score'], train_data2['gt_edge_score'], tau=0)

        return node_pos_loss + node_neg_loss + edge_pos_loss + edge_neg_loss, \
               node_pos_loss + node_neg_loss, edge_pos_loss + edge_neg_loss

    def balanced_MSE_loss(self, out, gt, tau):
        neg_idx = (gt[:, :] <= tau).nonzero()[:, 0]
        pos_idx = (gt[:, :] > tau).nonzero()[:, 0]

        if neg_idx.shape[0] == 0:
            neg_loss = torch.FloatTensor([0]).cuda()
        else:
            neg_loss = self.mse_loss(out[neg_idx, :], gt[neg_idx, :])
        if pos_idx.shape[0] == 0:
            pos_loss = torch.FloatTensor([0]).cuda()
        else:
            pos_loss = self.mse_loss(out[pos_idx, :], gt[pos_idx, :])
        return pos_loss, neg_loss
