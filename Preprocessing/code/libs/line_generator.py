import cv2
import torch
import torch.nn.functional as F

from libs.modules import *
from libs.utils import *

class generate_line(object):

    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.dataloader = dict_DB['dataloader']
        self.visualize = dict_DB['visualize']
        self.height = cfg.height
        self.width = cfg.width
        self.size = np.float32([cfg.height, cfg.width])

        # set outlier
        self.thresd1 = 200  # threshold of length
        self.thresd2 = 20   # threshold of distance from image boundary
        self.da = 1
        self.dd = 1

        self.scale_factor = self.cfg.scale_factor

        self.sampling_mode = self.cfg.sampling_mode
        self.min_angle_error = self.cfg.min_angle_error
        self.min_dist_error = self.cfg.min_dist_error

        self.center = np.array([(self.width - 1) / 2, (self.height - 1) / 2])
        self.max_dist = self.width // 2 - self.thresd2
        self.result = []


    def convert_to_line(self, angle, dist):
        a = np.tan(angle / 180 * math.pi)

        if angle != -90:
            b = -1
            c = dist * np.sqrt(a ** 2 + b ** 2) - (a * self.center[0] + b * self.center[1])

        else:
            a = 1
            b = 0
            c = self.center[0] + dist
        line_pts = find_endpoints_from_line_eq(line_eq=[a, b, c], size=[self.width - 1, self.height - 1])
        return line_pts

    def get_line_parameters(self, candidates):
        line_eq, check = line_equation(candidates)
        angle = transform_theta_to_angle(line_eq)
        dist = calculate_distance_from_center(line_eq, check, candidates, self.center)[:, 0, 0]
        return angle, dist

    def candidate_lines(self, mode='pri'):

        temp = {'endpts': [], 'angle': [], 'dist': [], 'idx': 0}

        if mode == 'pri':
            angle_step = 3
            dist_step = 5
        else:
            angle_step = 5
            dist_step = 10

        angle_list = [i for i in range(-90, 90, angle_step)]
        dist_list1 = [i for i in range(0, int(self.max_dist), dist_step)]
        dist_list2 = [-1 * dist_list1[i] for i in range(0, len(dist_list1))]
        dist_list = sorted(dist_list1 + dist_list2[1:])

        hough_space = {'idx': [], 'angle_list': [], 'dist_list': [], 'angle_step': [], 'dist_step': []}
        hough_space['idx'] = -1 * np.ones((len(angle_list), len(dist_list)), dtype=np.int32)
        hough_space['angle_list'] = angle_list
        hough_space['dist_list'] = dist_list
        hough_space['angle_step'] = angle_step
        hough_space['dist_step'] = dist_step
        hough_space['height'] = len(angle_list) * 3
        hough_space['width'] = len(dist_list)

        k = 0
        for i in range(len(angle_list)):
            for j in range(len(dist_list)):
                endpts = self.convert_to_line(angle_list[i], dist_list[j])
                check = candidate_line_filtering(pts=endpts,
                                                 size=(self.height, self.width),
                                                 thresd_boundary=self.thresd2,
                                                 thresd_length=self.thresd1)

                if check == 0:
                    temp['endpts'].append(np.expand_dims(endpts, axis=0))
                    temp['angle'].append(angle_list[i])
                    temp['dist'].append(dist_list[j])

                    hough_space['idx'][i, j] = k
                    k += 1

        hough_space['idx'] = np.concatenate((np.flip(hough_space['idx'], 1),
                                             hough_space['idx'],
                                             np.flip(hough_space['idx'], 1)), axis=0)

        candidates = np.float32(np.concatenate(temp['endpts']))
        print('The number of candidate lines %d' % candidates.shape[0])
        angle = np.float32(temp['angle'])
        dist = np.float32(temp['dist'])

        line_mask = generate_line_mask(line_pts=candidates / self.scale_factor,
                                       size=np.int32(self.size / self.scale_factor))

        if self.cfg.display == True:
            self.visualize.show['img_name'] = 'candidates.jpg'
            self.visualize.show['zero'] = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.visualize.draw_lines_cv(data=candidates, name='candidates', ref_name='zero', s=1)
            self.visualize.display_saveimg(dir_name=self.cfg.dir['out'], list=['candidates'])
        if mode == 'mul':
            self.mul_candidates = to_tensor(candidates)
            self.mul_line_mask = to_tensor(line_mask)
            self.mul_angle, self.mul_dist = to_tensor(angle), to_tensor(dist)

            self.mul_hough_space = hough_space

            h, w = hough_space['idx'].shape
            self.mul_hough_space_idx = to_tensor(hough_space['idx'][h//3:h//3*2, :])

    def get_approximated_mul_lines(self):

        result = {'mul': {'pts': [], 'angle_error': [], 'dist_error': [], 'region_idx': [], 'diff_idx': [],
                          'angle_idx': [], 'dist_idx': [], 'angle': [], 'dist': [],
                          'angle_offset': [], 'dist_offset': []},
                  'exist': []}
        temp_mul = []
        temp_mul_center = []
        temp_mul_angle_offset = []
        temp_mul_dist_offset = []
        temp_mul_idx = []

        gt_angle, gt_dist = self.get_line_parameters(self.mul_gt)
        for i in range(gt_angle.shape[0]):
            if np.round(to_np(gt_angle[i]), -1) == 90 and (self.mul_gt[i][0] + self.mul_gt[i][2]) / 2 < self.center[0] and gt_dist[i] < 0:
                gt_dist[i] = gt_dist[i] * -1
            elif np.round(to_np(gt_angle[i]), -1) == 90 and (self.mul_gt[i][0] + self.mul_gt[i][2]) / 2 > self.center[0] and gt_dist[i] > 0:
                gt_dist[i] = gt_dist[i] * -1
            elif np.round(to_np(gt_angle[i]), -1) == -90 and (self.mul_gt[i][0] + self.mul_gt[i][2]) / 2 < self.center[0] and gt_dist[i] > 0:
                gt_dist[i] = gt_dist[i] * -1
            elif np.round(to_np(gt_angle[i]), -1) == -90 and (self.mul_gt[i][0] + self.mul_gt[i][2]) / 2 > self.center[0] and gt_dist[i] < 0:
                gt_dist[i] = gt_dist[i] * -1

        angle_list = to_tensor(np.float32(self.mul_hough_space['angle_list']))
        dist_list = to_tensor(np.float32(self.mul_hough_space['dist_list']))


        self.angle_vx = torch.ones(angle_list.shape[0]).cuda()
        self.angle_vy = torch.tan(angle_list / 180 * math.pi)
        self.angle_vx[0] = 0
        self.angle_vy[0] = 1

        check = np.ones(self.mul_gt.shape[0], dtype=np.int32)

        for i in range(self.mul_gt.shape[0]):

            result['mul']['pts'].append(to_np(self.mul_gt[i]).reshape(1, 4))
            result['mul']['angle'].append(to_np(gt_angle[i]))
            result['mul']['dist'].append(to_np(gt_dist[i]))

            vx = torch.FloatTensor([1])[0].cuda()
            vy = torch.tan(gt_angle[i] / 180 * math.pi)
            if gt_angle[i] == -90:
                vx = torch.FloatTensor([0])[0].cuda()
                vy = torch.FloatTensor([1])[0].cuda()

            angle1 = self.compute_inner_product(self.angle_vx, self.angle_vy, vx, vy)
            angle2 = 180 - angle1
            angle_err = torch.min(angle1, angle2)

            if torch.argmin(angle_err) == 0:
                self.check_vertical = 1
            else:
                self.check_vertical = 0

            if torch.argmin(angle_err) == 0:

                if gt_angle[i] > 0 and gt_dist[i] > 0:
                    dist_err = torch.abs(dist_list - gt_dist[i] * -1)
                elif gt_angle[i] > 0 and gt_dist[i] < 0:
                    dist_err = torch.abs(dist_list - gt_dist[i] * -1)
                else:
                    dist_err = torch.abs(dist_list - gt_dist[i])
            else:
                dist_err = torch.abs(dist_list - gt_dist[i])
            if dist_err[torch.argmin(dist_err)] > self.mul_hough_space['dist_step']:

                check[i] = 0
                continue

            angle_idx = to_np(torch.argmin(angle_err) + self.mul_hough_space['height'] // 3)
            dist_dix = to_np(torch.argmin(dist_err))

            # sampling

            if self.sampling_mode == 'grid':
                # compute edge score
                center_idx = self.mul_hough_space['idx'][angle_idx, dist_dix]

                idx, diff_idx = self.adjacent_grid_idx(angle_idx, dist_dix,
                                                       self.da, self.dd,
                                                       self.mul_hough_space['idx'])

                offset_a, offset_d = self.adjacent_grid_offset(idx, gt_angle, gt_dist, angle_list, dist_list, i)

                if torch.sum(torch.isnan(offset_a)) == True:
                    print('nan error')
            result['mul']['angle_idx'].append(to_np(torch.argmin(angle_err) + self.mul_hough_space['height'] // 3))
            result['mul']['dist_idx'].append(to_np(torch.argmin(dist_err)))

            result['mul']['diff_idx'].append(to_np(diff_idx))
            result['mul']['angle_offset'].append(np.round(to_np(offset_a), 4))
            result['mul']['dist_offset'].append(to_np(offset_d))
            result['mul']['region_idx'].append(to_np(idx))

            temp_mul.append(to_np(self.mul_candidates[idx]).reshape(idx.shape[0], 4))
            temp_mul_center.append(to_np(self.mul_candidates[center_idx]).reshape(1, 4))
            temp_mul_angle_offset.append(np.round(to_np(offset_a), 4))
            temp_mul_dist_offset.append(to_np(offset_d))
            temp_mul_idx.append(to_np(idx))

            if np.max(to_np(offset_a)) > self.max_offset_a:
                self.max_offset_a = np.max(np.round(to_np(offset_a), 4))
            if np.max(to_np(offset_d)) > self.max_offset_d:
                self.max_offset_d = np.max(to_np(offset_d))
            if np.min(to_np(offset_a)) < self.min_offset_a:
                self.min_offset_a = np.min(np.round(to_np(offset_a), 4))
            if np.min(to_np(offset_d)) < self.min_offset_d:
                self.min_offset_d = np.min(to_np(offset_d))

        result['exist'] = check
        if temp_mul == []:
            print('no multiple line')
            return None

        else:
            temp_mul = np.concatenate(temp_mul)
            temp_mul_center = np.concatenate(temp_mul_center)


        if self.cfg.display == True:
            self.visualize.draw_lines_cv(data=temp_mul, name='gtlines')
            self.visualize.draw_lines_cv(data=temp_mul_center, name='overlap_center', ref_name='overlap')
            self.visualize.draw_lines_cv(data=temp_mul, name='overlap', ref_name='overlap')
            self.visualize.draw_lines_cv(data=temp_mul, name='overlap', ref_name='overlap')

            if self.cfg.display_offset_validation == True:
                if temp_mul != []:
                    line_pts = []
                    for i in range(len(temp_mul_idx)):

                        reg_angle = to_np(self.mul_angle)[temp_mul_idx[i]] + temp_mul_angle_offset[i]
                        reg_angle2 = reg_angle.copy()
                        reg_angle2[reg_angle < -90] += 180
                        reg_angle2[reg_angle > 90] -= 180
                        dist_list = to_np(self.mul_dist)[temp_mul_idx[i]].copy()
                        dist_list[reg_angle < -90] *= -1
                        dist_list[reg_angle > 90] *= -1
                        temp_mul_dist_offset[i][reg_angle < -90] = 0
                        temp_mul_dist_offset[i][reg_angle > 90] = 0
                        reg_dist2 = dist_list + temp_mul_dist_offset[i]

                        for j in range(reg_angle.shape[0]):
                            out_pts = self.convert_to_line(reg_angle2[j], reg_dist2[j])
                            if out_pts.shape[0] == 4:
                                line_pts.append(out_pts)
                    line_pts = np.float32(line_pts)
                    self.visualize.draw_lines_cv(data=line_pts, name='offset')
        return result

    def adjacent_grid_offset(self, region_idx, gt_angle, gt_dist, angle_list, dist_list, data_idx):
        temp_angle_err = torch.FloatTensor([]).cuda()
        temp_dist_err = torch.FloatTensor([]).cuda()
        for i in range(region_idx.shape[0]):
            idx = (self.mul_hough_space_idx == region_idx[i]).nonzero()[0]

            x1 = self.angle_vx[idx[0]]
            x2 = torch.FloatTensor([1])[0].cuda()
            y1 = self.angle_vy[idx[0]]
            y2 = torch.tan(gt_angle[data_idx] / 180 * math.pi)

            if gt_angle[data_idx] == -90:
                x2 = torch.FloatTensor([0])[0].cuda()
                y2 = torch.FloatTensor([1])[0].cuda()
            angle1 = self.compute_inner_product(x1, y1, x2, y2)
            if torch.isnan(angle1) == True:
                angle1 = torch.FloatTensor([0])[0].cuda()
            angle2 = 180 - angle1
            angle_err = torch.min(angle1, angle2)

            # angle error

            flag = 0
            flag2 = 0
            if angle_list[idx[0]] < 0 and gt_angle[data_idx] > 0:
                if torch.abs(angle_list[idx[0]] - gt_angle[data_idx]) > 90:
                    flag = -1
                    flag2 = 1
                else:
                    flag = 1
            elif angle_list[idx[0]] <= 0 and gt_angle[data_idx] <= 0:
                if angle_list[idx[0]] - gt_angle[data_idx] > 0:
                    flag = -1
                else:
                    flag = 1
            elif angle_list[idx[0]] >= 0 and gt_angle[data_idx] >= 0:
                if angle_list[idx[0]] - gt_angle[data_idx] > 0:
                    flag = -1
                else:
                    flag = 1
            elif angle_list[idx[0]] > 0 and gt_angle[data_idx] < 0:
                if torch.abs(angle_list[idx[0]] - gt_angle[data_idx]) > 90:
                    flag = 1
                    flag2 = 1
                else:
                    flag = -1
            if flag == 0:
                print('strong angle error')
            angle_err *= flag

            dist_err = gt_dist[data_idx] - dist_list[idx[1]]
            if flag2 == 1:
                dist_err = 0

            temp_angle_err = torch.cat((temp_angle_err, torch.FloatTensor([angle_err]).cuda()))
            temp_dist_err = torch.cat((temp_dist_err, torch.FloatTensor([dist_err]).cuda()))

        return temp_angle_err, temp_dist_err

    def compute_inner_product(self, x1, y1, x2, y2):
        a = ((x1 * x2 + y1 * y2) / (torch.sqrt(x1 * x1 + y1 * y1) * torch.sqrt(x2 * x2 + y2 * y2)))
        b = torch.acos(a) / math.pi * 180
        return b

    def adjacent_grid_idx(self, a_idx, d_idx, da, dd, hough_space_idx):
        h, w = hough_space_idx.shape
        x1 = np.maximum(a_idx - da, 0)
        x2 = np.minimum(a_idx + da + 1, h)
        y1 = np.maximum(d_idx - dd, 0)
        y2 = np.minimum(d_idx + dd + 1, w)

        dx = x2 - x1
        dy = y2 - y1
        X, Y = torch.meshgrid(torch.arange(x1, x2).cuda(),
                              torch.arange(y1, y2).cuda())
        X = to_tensor(a_idx) - X.view(dx, dy, 1)
        Y = to_tensor(d_idx) - Y.view(dx, dy, 1)
        diff_idx = torch.cat((X, Y), dim=2).view(-1, 2)
        region_idx = to_tensor(hough_space_idx[x1:x2, y1:y2]).type(torch.long).reshape(-1)

        diff_idx = diff_idx[(region_idx != -1).nonzero()[:, 0]]  # exclude idx value -1
        region_idx = region_idx[(region_idx != -1).nonzero()][:, 0]  # exclude idx value -1

        return region_idx, diff_idx

    def run(self):
        print('start')
        self.candidate_lines(mode='mul')

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='candidates', data=to_np(self.mul_candidates))
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='angle', data=to_np(self.mul_angle))
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='dist', data=to_np(self.mul_dist))
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='hough_space', data=self.mul_hough_space)
        datalist = []

        self.max_offset_a = -9999
        self.max_offset_d = -9999
        self.min_offset_a = 9999
        self.min_offset_d = 9999
        for i, batch in enumerate(self.dataloader):
            self.img = batch['img'].cuda()

            self.mul_gt = batch['label']['mul_gt'][0].cuda()
            self.img_name = batch['img_name'][0]

            result = []
            for j in range(0, 2):  # 1: horizontal flip

                if j == 1 and self.cfg.data_flip == False:
                    continue
                if j == 1:
                    self.img = self.img.flip(3)
                    self.mul_gt[:, 0] = self.width - 1 - self.mul_gt[:, 0]
                    self.mul_gt[:, 2] = self.width - 1 - self.mul_gt[:, 2]

                self.visualize.update_image(self.img[0])
                self.visualize.update_image_name(self.img_name)

                self.visualize.show['gtlines'] = np.copy(self.visualize.show['img'])
                self.visualize.show['label'] = np.copy(self.visualize.show['img'])
                self.visualize.draw_lines_cv(data=self.mul_gt, name='label', ref_name='label',
                                             s=2, color=(0, 0, 255))
                self.visualize.show['overlap'] = np.copy(self.visualize.show['label'])

                result_mul = self.get_approximated_mul_lines()
                result.append(result_mul)

            # save data
            if self.cfg.display == True and result_mul != None:
                self.visualize.display_saveimg(dir_name=self.cfg.dir['out'] + 'display/',
                                               list=['img', 'overlap_center', 'gtlines', 'overlap', 'offset'])
            if self.cfg.save_pickle == True and result_mul != None:
                save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name=self.img_name.replace('.jpg', ''), data=result)
                datalist.append(self.img_name)
            print('image %d ===> %s clear' % (i, self.img_name))
            print(self.max_offset_a, self.max_offset_d, self.min_offset_a, self.min_offset_d)

        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + 'pickle/', file_name='datalist', data=datalist)

        print(self.max_offset_a, self.max_offset_d, self.min_offset_a, self.min_offset_d)