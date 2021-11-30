import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from libs.utils import *
from libs.modules import *

class Evaluation_HIoU(nn.Module):
    def __init__(self, cfg):
        super(Evaluation_HIoU, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        # candidates
        self.size = to_tensor(np.float32(self.cfg.size))
        self.line_num = np.zeros(2)

        # generate grid
        self.grid = {}
        self.X = {}
        self.Y = {}
        self.generate_grid(self.cfg.sf_hiou, self.height // self.cfg.sf_hiou, self.width // self.cfg.sf_hiou)

        self.color_list = np.array([[[0, 0, 255],
                                    [0, 255, 0],
                                    [255, 0, 0],
                                    [0, 255, 255],
                                    [255, 0, 255],
                                    [255, 255, 0],
                                    [255, 200, 128],
                                    [128, 255, 200],
                                    [200, 128, 255],
                                    [255, 128, 200],
                                    [200, 255, 128],
                                    [128, 200, 255],
                                    [255, 255, 255],
                                    [255, 153, 0],
                                    [255, 51, 102]]])

        self.dist = 3

    def generate_grid(self, sf, height, width):
        X, Y = np.meshgrid(np.linspace(0, width - 1, width),
                           np.linspace(0, height - 1, height))

        self.X[sf] = torch.tensor(X, dtype=torch.float, requires_grad=False).cuda()
        self.Y[sf] = torch.tensor(Y, dtype=torch.float, requires_grad=False).cuda()
        self.grid[sf] = torch.cat((self.X[sf].view(1, height, width, 1),
                                   self.Y[sf].view(1, height, width, 1)), dim=3)


    def line_equation(self, mode=None):

        if mode == 'norm':
            data = self.line_pts_norm.clone()
        else:
            data = self.line_pts.clone()
            data = data.cuda().type(torch.float32)

        # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
        line_eq = torch.zeros((data.shape[0], 3)).cuda()
        line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
        line_eq[:, 1] = -1
        line_eq[:, 2] = (-1 * line_eq[:, 0] * data[:, 0] + data[:, 1])
        check = ((data[:, 0] - data[:, 2]) == 0)

        return line_eq, check

    def calculate_distance(self, line_eq, check):  # line-point distance

        num = line_eq.shape[0]
        a = line_eq[:, 0].view(num, 1, 1)
        b = line_eq[:, 1].view(num, 1, 1)
        c = line_eq[:, 2].view(num, 1, 1)

        dist = (self.grid[self.sf][:, :, :, 0] * a + self.grid[self.sf][:, :, :, 1] * b + c) / \
               torch.sqrt(a * a + b * b)

        if True in check:
            dist[check == True] = (self.grid[self.sf][:, :, :, 0].cuda().type(torch.float32) -
                                   self.line_pts[check == True, 0].view(-1, 1, 1).cuda().type(torch.float32))
        self.dist_map = dist

    def generate_region_mask(self):
        b, h, w = self.dist_map.shape
        region1 = (0 < self.dist_map).view(b, h, w, 1)
        region2 = (self.dist_map < 0).view(b, h, w, 1)

        return torch.cat((region1, region2), dim=3)

    def update(self, line_pts, scale_factor):

        self.line_pts = line_pts
        self.sf = scale_factor

    def update_dataset_name(self, dataset_name):

        self.dataset_name = dataset_name

    def preprocess(self, line_pts):

        output = {'region_mask': {},
                  'line_mask': {},
                  'grid': {},
                  'weight': {}}

        self.update(line_pts, self.cfg.sf_hiou)
        line_eq, check = self.line_equation()
        self.calculate_distance(line_eq, check)
        region_mask = self.generate_region_mask()

        if self.cfg.disp_hiou == True:

            if region_mask.shape[0] != 0:
                self.visualize_separate_region(mask=region_mask[0],
                                               line_pts=line_pts[0],
                                               dir_name=self.cfg.dir['out'] + 'HIoU/vis_check/region_mask/',
                                               file_name='region.jpg')
        output['region_mask'] = region_mask

        return output


    def measure_IoU(self, X1, X2):
        X = X1 + X2

        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)

        iou = X_inter / X_uni

        return iou

    def measure_hiou_metric(self, pred, gt):

        m = len(pred)
        n = len(gt)
        score_table = torch.zeros((m, n), dtype=torch.float).cuda()

        for i in range(m):
            score_table[i, :] = self.measure_IoU(pred[i:i+1], gt)

        result = 0

        for i in range(m):
            result += score_table[i].max()
        score_table_T = score_table.T
        for j in range(n):
            result += score_table_T[j].max()
        result = result / (m + n)

        return result


    def generate_inter_region_mask(self, region_mask):

        memory_region_mask = region_mask[:1].clone()

        for i in range(1, region_mask.shape[0]):
            temp_region_mask = torch.BoolTensor([]).cuda()
            for j in range(2):
                region = memory_region_mask * region_mask[i:i+1, :, :, j:j+1]
                temp_region_mask = torch.cat((temp_region_mask, region), dim=3)
            memory_region_mask = temp_region_mask.clone()

        memory_region_mask = memory_region_mask[0].permute(2, 0, 1)
        area = torch.sum(memory_region_mask, dim=(1, 2))
        idx = (area != 0).nonzero()[:, 0]
        piece_mask = memory_region_mask[idx]

        return piece_mask.type(torch.float)

    def visualize_separate_region(self, mask, line_pts, dir_name, file_name):


        mask1 = np.uint8(to_3D_np(to_np(mask[:, :, 0])) * 255)
        mask1[:, :, [0, 1]] = 0
        mask2 = np.uint8(to_3D_np(to_np(mask[:, :, 1])) * 255)
        mask2[:, :, [0, 2]] = 0

        disp_mask = mask1 + mask2

        mask = cv2.resize(disp_mask, (self.width, self.height))

        pt_1 = (line_pts[0], line_pts[1])
        pt_2 = (line_pts[2], line_pts[3])

        mask = cv2.line(mask, pt_1, pt_2, (255, 255, 255), 3)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, mask)


    def run(self, pred_pts, gt_pts, img_name):

        result = dict()
        self.img_name = img_name

        if pred_pts.shape[0] != 0:
            pred_mask = self.preprocess(pred_pts)
            gt_mask = self.preprocess(gt_pts)
            self.line_num[:] = pred_pts.shape[0], gt_pts.shape[0]
            pred_inter_mask = self.generate_inter_region_mask(pred_mask['region_mask'])
            gt_inter_mask = self.generate_inter_region_mask(gt_mask['region_mask'])

            result['IOU'] = self.measure_hiou_metric(pred_inter_mask, gt_inter_mask)

            if self.cfg.disp_hiou == True:
                self.visualize_inter_region(pred_inter_mask, gt_inter_mask)
                self.visualize_result(pred_pts, gt_pts, result)

        else:
            result['IOU'] = "not_detection_image"

        return result


    def visualize_result(self, pred_line, gt_line, output):

        pred_img = cv2.imread(self.cfg.dir['dataset'][self.dataset_name + '_img'] + self.img_name)
        gt_img = cv2.imread(self.cfg.dir['dataset'][self.dataset_name + '_img'] + self.img_name)

        pred_img = cv2.resize(pred_img, (self.cfg.width, self.cfg.height))
        gt_img = cv2.resize(gt_img, (self.cfg.width, self.cfg.height))

        pred_line = to_np(pred_line)
        for i in range(pred_line.shape[0]):
            pts = pred_line[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            pred_img = cv2.line(pred_img, pt_1, pt_2, (255, 0, 0), 2)


        cv2.putText(pred_img, "IOU : {:01.4f}".format(output['IOU']), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        gt_line = to_np(gt_line)
        for i in range(gt_line.shape[0]):
            pts = gt_line[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            gt_img = cv2.line(gt_img, pt_1, pt_2, (255, 0, 0), 2)

        blank = np.full((self.cfg.height, 5, 3), 255, dtype=np.int32)
        total = np.concatenate((pred_img, blank, gt_img), axis=1)

        file_dir = self.cfg.dir['out'] + 'HIoU/vis/'
        mkdir(file_dir)

        cv2.imwrite(file_dir + self.img_name + '_total.jpg', total)


    def visualize_inter_region(self, pred, gt):
        m = len(pred)
        n = len(gt)
        pred_vis = np.zeros((self.cfg.width, self.cfg.height, 3))
        gt_vis = np.zeros((self.cfg.width, self.cfg.height, 3))

        pred = to_np(pred)
        gt = to_np(gt)
        for i in range(m):
            if i >= self.color_list.shape[1]:
                continue

            tmp = pred[i].reshape((self.cfg.width, self.cfg.height, 1))
            tmp = np.repeat(tmp, 3, axis=2)
            tmp = tmp * self.color_list[:, i, :]
            pred_vis += tmp

        for j in range(n):
            if j >= self.color_list.shape[1]:
                continue
            tmp = gt[j].reshape((self.cfg.width, self.cfg.height, 1))
            tmp = np.repeat(tmp, 3, axis=2)
            tmp = tmp * self.color_list[:, j, :]
            gt_vis += tmp

        blank = np.full((self.cfg.height, 5, 3), 255, dtype=np.int32)

        total = np.concatenate((pred_vis, blank, gt_vis), axis=1)

        file_dir = self.cfg.dir['out'] + 'HIoU/vis_segmap/'
        mkdir(file_dir)

        cv2.imwrite(file_dir + self.img_name + '_total.jpg', total)

