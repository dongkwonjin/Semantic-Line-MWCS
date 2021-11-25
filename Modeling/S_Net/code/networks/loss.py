import numpy as np

import torch
import torch.nn as nn

class Loss_Function(nn.Module):
    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_score = nn.MSELoss()

        self.weight = 1


    def forward(self, out, gt_score, gt_offset):
        l_score = self.balanced_BCE_loss(out['score'], gt_score)
        l_off_a = self.balanced_MSE_loss(out['offset'][0][:, :1], gt_offset[0][:, :1], gt_score)
        l_off_d = self.balanced_MSE_loss(out['offset'][0][:, 1:], gt_offset[0][:, 1:], gt_score)

        return {'sum': l_score + l_off_a + l_off_d,
                'off_a': l_off_a,
                'off_d': l_off_d,
                'score': l_score}

    def balanced_BCE_loss(self, out, gt):
        neg_idx = (gt[:, :] == 0).nonzero()[:, 1]
        pos_idx = (gt[:, :] != 0).nonzero()[:, 1]

        neg_loss = self.loss_bce(out[:, 0, neg_idx], gt[:, neg_idx])
        if pos_idx.shape[0] == 0:
            return neg_loss * self.weight
        else:
            pos_loss = self.loss_bce(out[:, 0, pos_idx], gt[:, pos_idx])
            return pos_loss + neg_loss * self.weight

    def balanced_MSE_loss(self, out, gt, gt_score):
        neg_idx = (gt_score[:, :] == 0).nonzero()[:, 1]
        pos_idx = (gt_score[:, :] != 0).nonzero()[:, 1]

        neg_loss = self.loss_mse(out[neg_idx, :], gt[neg_idx, :])
        if pos_idx.shape[0] == 0:
            return neg_loss * self.weight
        else:
            pos_loss = self.loss_mse(out[pos_idx, :], gt[pos_idx, :])
            return pos_loss + neg_loss * self.weight
