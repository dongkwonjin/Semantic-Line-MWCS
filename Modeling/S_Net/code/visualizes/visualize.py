import cv2

import matplotlib.pyplot as plt

from libs.utils import *
from libs.modules import *
import cv2
import math
import numpy as np

from libs.utils import *

class Visualize_cv(object):

    def __init__(self, cfg):

        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

        # result
        self.center = np.array(self.cfg.center_pt)
        self.max_dist = self.cfg.max_dist
        self.top_k = 20
        self.minimum_score = 0.5
        # load
        self.candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.cand_angle = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'angle'))
        self.cand_dist = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'dist'))

    def update_org_image(self, path, name='org_img'):
        img = cv2.imread(path)
        self.show[name] = img

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_label(self, label, name='label'):
        label = to_np(label)
        label = np.repeat(np.expand_dims(np.uint8(label != 0) * 255, axis=2), 3, 2)
        self.show[name] = label

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name

    def edge_map_to_rgb_image(self, data):
        data = np.repeat(np.uint8(to_np2(data.permute(1, 2, 0) * 255)), 3, 2)
        data = cv2.resize(data, (self.width, self.height))
        return data

    def draw_text(self, pred, label, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        cv2.rectangle(img, (1, 1), (250, 120), color, 1)
        cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, 'label : ' + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        self.show[name] = img

    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def draw_lane_points_cv(self, data_x, data_y, name, ref_name='org_img', color=(0, 255, 0), s=4):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(len(data_x)):
            for j in range(data_x[i].shape[0]):
                pts = (int(data_x[i][j]), int(data_y[i][j]))
                img = cv2.circle(img, pts, s, color, -1)

        self.show[name] = img

    def display_saveimg(self, dir_name, file_name, list):
        # boundary line
        if self.show[list[0]].shape[0] != self.line.shape[0]:
            self.line = np.zeros((self.show[list[0]].shape[0], 3, 3), dtype=np.uint8)
            self.line[:, :, :] = 255
        disp = self.line

        for i in range(len(list)):
            if list[i] not in self.show.keys():
                continue
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp)

    def display_for_train_reg(self, batch, out, idx):

        self.update_image(batch['img'][0])
        self.update_image_name(batch['img_name'][0])

        line_pts = {'out': [], 'out_score': [], 'out_mask': [], 'out_center': [], 'out_pos': [],
                    'gt_train': [], 'gt': []}



        # out high scored line
        idx_score = to_np((out['score'][0] > self.minimum_score).nonzero()[:, 1])
        line_pts['out_score'] = self.candidates[idx_score]

        # out hough line
        if idx_score.shape[0] != 0:
            line_pts['out'] = self.process_for_reg(idx_score,
                                                   out['offset'][0][idx_score, 0],
                                                   out['offset'][0][idx_score, 1])
        else:
            line_pts['out'] = np.array([])

        idx_pos = to_np((batch['score'] != 0).nonzero()[:, 1])
        if idx_pos.shape[0] != 0:
            line_pts['out_pos'] = self.process_for_reg(idx_pos,
                                                   out['offset'][0][idx_pos, 0],
                                                   out['offset'][0][idx_pos, 1])
        else:
            line_pts['out_pos'] = np.array([])

        # gt train line
        idx_gt_pos = to_np((batch['offset'] != 0).nonzero()[:, 1])

        if idx_gt_pos.shape[0] != 0:
            line_pts['gt_train'] = self.process_for_reg(idx_gt_pos,
                                                   out['offset'][0][idx_gt_pos, 0],
                                                   out['offset'][0][idx_gt_pos, 1])
        else:
            line_pts['gt_train'] = np.array([])

        # to numpy
        line_pts['out'] = np.float32(line_pts['out'])
        line_pts['out_pos'] = np.float32(line_pts['out_pos'])
        line_pts['gt_train'] = np.float32(line_pts['gt_train'])
        line_pts['gt'] = np.float32(batch['mul_gt'][0])

        # draw lines

        self.draw_lines_cv(data=line_pts['out'], name='out', color=(255, 0, 0))
        self.draw_lines_cv(data=line_pts['out_score'], name='out_score')
        self.draw_lines_cv(data=line_pts['out_pos'], name='out_pos')
        self.draw_lines_cv(data=line_pts['gt'], name='gt')
        self.draw_lines_cv(data=line_pts['gt_train'], name='gt_train', color=(255, 0, 0))

        # save result
        self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/',
                             file_name=str(idx) + '.jpg',
                             list=['img', 'out_score', 'out', 'out_pos', 'gt_train', 'gt'])

    def display_for_test(self, out, mul_gt, idx, mode, dataset_name):

        line_pts = {'out': [], 'out_score': [], 'out_reg': [], 'out_mask': [], 'gt': []}

        # high score line
        idx_score = (out['score'][0] > self.minimum_score).nonzero()[:, 1]
        idx_score = to_np(idx_score)
        line_pts['out_score'] = self.candidates[idx_score]

        # out hough line

        # draw lines
        self.draw_lines_cv(data=line_pts['out_score'], name='out_score')
        self.draw_lines_cv(data=out['out_pts_cls'], name='out_cls')
        self.draw_lines_cv(data=out['out_pts_reg'], name='out_reg')
        self.draw_lines_cv(data=np.expand_dims(out['out_pts_cls'][0], 0), name='out_cls', ref_name='out_cls', color=(0, 255, 0))
        self.draw_lines_cv(data=np.expand_dims(out['out_pts_reg'][0], 0), name='out_reg', ref_name='out_reg', color=(0, 255, 0))
        self.draw_lines_cv(data=mul_gt, name='gt', color=(255, 0, 0))

        self.display_saveimg(dir_name=self.cfg.dir['out'] + mode + '_' + dataset_name + '/display/',
                             file_name=str(idx) + '.jpg',
                             list=['img', 'out_score', 'out_cls', 'out_reg', 'gt'])

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


    def process_for_reg(self, idx, angle_offset, dist_offset):
        cls_angle = self.cand_angle[idx]
        cls_dist = self.cand_dist[idx]

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
        line_pts = np.float32(line_pts)

        return line_pts