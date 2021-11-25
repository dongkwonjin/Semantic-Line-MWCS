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
        hough_space = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'hough_space')
        self.hough_h, self.hough_w = hough_space['idx'].shape
        self.angle_list = to_tensor(np.float32(hough_space['angle_list'])) / 90
        self.dist_list = to_tensor(np.float32(hough_space['dist_list'])) / self.max_dist

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

    def display_for_train_node_score(self, batch, out, train_data, idx):

        self.update_image(batch['img'][0])
        self.update_image_name(batch['img_name'][0])

        for i in range(0, train_data['node_idx_set'].shape[0], 3):

            line_pts = {'out': [], 'node': [], 'gt': []}

            # out masked line
            node_idx = to_np(train_data['node_idx_set'][i:i+1, 0])
            out_pts = self.candidates[node_idx, :]
            line_pts['node'].append(out_pts)

            # to numpy
            line_pts['gt'] = np.float32(batch['mul_gt'][0])
            line_pts['node'] = np.concatenate(line_pts['node'])

            # draw lines
            self.draw_lines_cv(data=line_pts['gt'], name='gt')
            self.draw_lines_cv(data=line_pts['node'], name='node')

            self.draw_text(pred=np.round(to_np2(out['node_score'][i, 0]), 3),
                           label=np.round(to_np2(train_data['gt_node_score'][i, 0]), 3),
                           name='node', ref_name='node', color=(0, 255, 0))


            # save result
            self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/',
                                 file_name=str(idx) + '_' + str(i) + '_node.jpg',
                                 list=['img', 'node', 'gt'])
            if self.cfg.disp_train_per_case == True:
                self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/' + str(int(train_data['node_case'][i])) + '/',
                                     file_name=str(idx) + '_' + str(i) + '_node.jpg',
                                     list=['img', 'node', 'gt'])


    def display_for_train_edge_score(self, batch, out, train_data, idx):

        self.update_image(batch['img'][0])
        self.update_image_name(batch['img_name'][0])


        for i in range(0, train_data['edge_idx_set'].shape[0], 3):

            line_pts = {'out': [], 'edge': [], 'gt': []}

            # out masked line
            idx1 = to_np(train_data['edge_idx_set'][i:i+1, 0])
            out_pts = self.candidates[idx1, :]
            line_pts['edge'].append(out_pts)
            idx2 = to_np(train_data['edge_idx_set'][i:i+1, 1])
            out_pts = self.candidates[idx2, :]
            line_pts['edge'].append(out_pts)

            # to numpy
            line_pts['gt'] = np.float32(batch['mul_gt'][0])
            line_pts['edge'] = np.concatenate(line_pts['edge'])

            # draw lines
            self.draw_lines_cv(data=line_pts['gt'], name='gt')
            self.draw_lines_cv(data=line_pts['edge'], name='edge')

            self.draw_text(pred=np.round(to_np2(out['edge_score'][i, 0]), 3),
                           label=np.round(to_np2(train_data['gt_edge_score'][i, 0]), 3),
                           name='edge', ref_name='edge', color=(0, 255, 0))


            # save result
            self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/',
                                 file_name=str(idx) + '_' + str(i) + '_edge.jpg',
                                 list=['img', 'edge', 'gt'])
            if self.cfg.disp_train_per_case == True:
                self.display_saveimg(dir_name=self.cfg.dir['out'] + 'train/display/' + str(int(train_data['edge_case'][i])) + '/',
                                     file_name=str(idx) + '_' + str(i) + '_edge.jpg',
                                     list=['img', 'edge', 'gt'])



    def display_for_test(self, out, mul_gt, idx, mode, dataset_name):

        line_pts = {'out': [], 'out_score': [], 'out_mask': [], 'out_updated': [], 'out_updated_idx': [],
                    'gt': []}

        # high score line
        idx_score = (out['score'][0] > self.minimum_score).nonzero()[:, 1]
        idx_score = to_np(idx_score)
        line_pts['out_score'] = self.candidates[idx_score]

        # out center line
        out_pts = self.candidates[to_np(out['center_idx'])]
        line_pts['out_center'] = out_pts



        # to np
        if line_pts['out_mask'] == []:
            line_pts['out_mask'] = np.float32(line_pts['out_mask'])
        else:
            line_pts['out_mask'] = np.concatenate(line_pts['out_mask'])

        # draw lines
        self.draw_lines_cv(data=line_pts['out_center'], name='center')
        self.draw_lines_cv(data=line_pts['out_score'], name='out_score')
        self.draw_lines_cv(data=out['cls_mul_pts'], name='out_cls')

        if 'reg_mul_pts' in out.keys():
            self.draw_lines_cv(data=out['reg_mul_pts'], name='out_reg')

        self.draw_lines_cv(data=mul_gt, name='gt', color=(255, 0, 0))

        self.display_saveimg(dir_name=self.cfg.dir['out'] + mode + '_' + dataset_name + '/display/',
                             file_name=str(idx) + '.jpg',
                             list=['img', 'center', 'out_score', 'out_cls', 'out_reg', 'gt'])

    def convert_to_line(self, angle, dist):
        angle = to_np2(angle) * 90
        dist = to_np2(dist) * self.max_dist
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

class Visualize_graph(object):

    def __init__(self, cfg, mode=''):

        self.cfg = cfg
        self.mode = mode
        self.height = cfg.height
        self.width = cfg.width

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.param = {'linewidth': [0, 1, 2, 3, 4],
                      'color': ['yellow', 'red', 'lime']}

        self.show = {}

        self.candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')

    def update_image(self, img, name='img'):
        img = to_np(img.permute(1, 2, 0))
        img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.show[name] = img

    def update_image_name(self, img_name):
        self.show['img_name'] = img_name


    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def draw_text(self, pred, label, name, ref_name='img', color=(255, 0, 0)):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        cv2.rectangle(img, (1, 1), (250, 120), color, 1)
        cv2.putText(img, 'pred : ' + str(pred), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, 'label : ' + str(label), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
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

    def update_batch(self, batch, mode, dataset_name):
        self.mode = mode
        self.dataset_name = dataset_name
        self.batch = batch
        self.update_image(self.batch['img'][0])
        self.update_image_name(self.batch['img_name'][0])
        self.draw_lines_cv(data=self.batch['mul_gt'][0], name='gt', color=(255, 0, 0))
        self.draw_lines_cv(data=self.batch['pri_gt'][0], name='gt', ref_name='gt', color=(0, 255, 0))

    def display_post_process(self, out, visit, score, dir_name, file_name):


        self.draw_lines_cv(data=out[visit == 1],
                           name='post',
                           ref_name='img',
                           color=(0, 255, 0))

        self.draw_lines_cv(data=out[visit == 2],
                           name='post',
                           ref_name='post')

        self.draw_text(pred=score, label=1,
                       name='post', ref_name='post',
                       color=(0, 255, 0))

        self.display_saveimg(dir_name=dir_name,
                             file_name=file_name,
                             list=['img', 'post', 'gt'])

    def display_adj_similarity_map_for_vertex(self, out, dir_name, file_name):

        pair_idx_set = out['pair_idx_set']
        score = np.round(to_np(out['edge_score'][:, 0]), 3)

        for i in range(0, out['pair_idx_set'].shape[0], 3):
            if (pair_idx_set[i, 0] != pair_idx_set[i, 1]) == True:
                continue
            for k in range(4):

                sim_map = out['sim_map'][i]

                # sim_map
                out_map = self.Colormap(img=self.show['img'],
                                        mask=to_np(sim_map[k, 0]))
                self.show['sim_map'] = out_map

                self.draw_lines_cv(data=np.expand_dims(self.candidates[pair_idx_set[i, 0]], 0),
                                   name='sim_map',
                                   ref_name='sim_map',
                                   color=(0, 255, 0))
                self.draw_lines_cv(data=np.expand_dims(self.candidates[pair_idx_set[i, 1]], 0),
                                   name='sim_map',
                                   ref_name='sim_map',
                                   color=(0, 255, 0))
                self.draw_text(pred=score[i], label=1,
                               name='sim_map', ref_name='sim_map',
                               color=(0, 255, 0))

                self.display_saveimg(dir_name=dir_name,
                                     file_name=file_name + '_' + str(i) + '_' + str(k) + '.jpg',
                                     list=['img', 'sim_map', 'gt'])
            # break

    def display_adj_similarity_map_for_edge(self, out, dir_name, file_name):

        pair_idx_set = out['edge_idx_set']
        score = np.round(to_np(out['edge_score'][:, 0]), 3)

        if self.mode == 'test':
            step = 1
        else:
            step = 10

        for i in range(0, out['edge_idx_set'].shape[0], step):
            if (pair_idx_set[i, 0] == pair_idx_set[i, 1]) == True:
                continue
            for k in range(2):

                sim_map = out['sim_map'][i]

                # sim_map
                out_map = self.Colormap(img=self.show['img'],
                                        mask=to_np(sim_map[k, 0]))
                self.show['sim_map'] = out_map

                self.draw_lines_cv(data=np.expand_dims(self.candidates[pair_idx_set[i, k]], 0),
                                   name='sim_map',
                                   ref_name='sim_map',
                                   color=(0, 255, 0))
                # self.draw_lines_cv(data=np.expand_dims(self.candidates[pair_idx_set[i, 1]], 0),
                #                    name='sim_map',
                #                    ref_name='sim_map',
                #                    color=(0, 255, 0))
                self.draw_text(pred=score[i], label=1,
                               name='sim_map', ref_name='sim_map',
                               color=(0, 255, 0))

                self.display_saveimg(dir_name=dir_name,
                                     file_name=file_name + '_' + str(i) + '_' + str(k) + '.jpg',
                                     list=['img', 'sim_map', 'gt'])

    def display_edge_score(self, out, dir_name, file_name):

        pair_idx_set = out['edge_idx_set']
        score = np.round(to_np(out['edge_score'][:, 0]), 3)

        if self.mode == 'test':
            step = 1
        else:
            step = 10

        for i in range(0, out['edge_idx_set'].shape[0], step):
            if (pair_idx_set[i, 0] == pair_idx_set[i, 1]) == True:
                continue

            self.draw_lines_cv(data=self.candidates[to_np(pair_idx_set[i, :])],
                               name='edge_score',
                               ref_name='img',
                               color=(0, 255, 0))
            self.draw_text(pred=score[i], label=1,
                           name='edge_score', ref_name='edge_score',
                           color=(0, 255, 0))

            self.display_saveimg(dir_name=dir_name,
                                 file_name=file_name + '_' + str(i) + '.jpg',
                                 list=['img', 'edge_score', 'gt'])


    def Colormap(self, img, mask, alpha=0.5):

        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.uint8(cv2.resize(mask, (self.width, self.height)) * 255)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        out = img * alpha + mask * (1 - alpha)

        return out