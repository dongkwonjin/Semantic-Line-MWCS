import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils import *
from libs.modules import *

class Separated_Region(nn.Module):
    def __init__(self, cfg):
        super(Separated_Region, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.max_dist = self.cfg.max_dist

        # candidates
        self.size = to_tensor(np.float32(self.cfg.size))
        self.scale_factor = self.cfg.scale_factor
        self.dist = 5

        self.candidates = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates'))

        # generate grid
        self.grid = {}
        self.X = {}
        self.Y = {}
        for sf in self.scale_factor:
            self.generate_grid(sf, self.height // sf, self.width // sf)

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
                                    [128, 200, 255]]])

        self.dist = 3

    def generate_grid(self, sf, height, width):
        X, Y = np.meshgrid(np.linspace(0, width - 1, width),
                           np.linspace(0, height - 1, height))

        self.X[sf] = torch.tensor(X, dtype=torch.float, requires_grad=False).cuda()
        self.Y[sf] = torch.tensor(Y, dtype=torch.float, requires_grad=False).cuda()
        self.grid[sf] = torch.cat((self.X[sf].view(1, height, width, 1),
                                   self.Y[sf].view(1, height, width, 1)), dim=3)

    def generate_flipped_grid(self):
        num = self.line_pts.shape[0]

        h, w = self.height // self.sf, self.width // self.sf

        pts_norm = (self.line_pts / (self.size / self.sf - 1) - 0.5) * 2  # [-1, 1]
        self.line_pts_norm = pts_norm

        line_eq, check = self.line_equation(mode='norm')

        X, Y = np.meshgrid(np.linspace(-1, 1, w),
                           np.linspace(-1, 1, h))  # after x before

        grid_X0 = torch.tensor(X, dtype=torch.float, requires_grad=False).view(1, h, w).cuda()
        grid_Y0 = torch.tensor(Y, dtype=torch.float, requires_grad=False).view(1, h, w).cuda()
        grid_X0 = grid_X0.expand(num, h, w)
        grid_Y0 = grid_Y0.expand(num, h, w)

        line_eq = line_eq.view(num, 1, 1, 3)
        grid_X1, grid_Y1 = point_flip(line_eq, check, pts_norm, grid_X0, grid_Y0)

        flipped_grid = torch.cat((grid_X1.view(num, h, w, 1), grid_Y1.view(num, h, w, 1)), dim=3)
        return flipped_grid


    def line_equation(self, mode=None):

        if mode == 'norm':
            data = self.line_pts_norm.clone()
        else:
            data = self.line_pts.clone()

        # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
        line_eq = torch.zeros((data.shape[0], 3)).cuda()
        line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
        line_eq[:, 1] = -1
        line_eq[:, 2] = -1 * line_eq[:, 0] * data[:, 0] + data[:, 1]
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
            dist[check == True] = (self.grid[self.sf][:, :, :, 0] -
                                   self.line_pts[check == True, 0].view(-1, 1, 1))

        self.dist_map = dist

    def generate_region_mask(self):
        b, h, w = self.dist_map.shape
        region1 = (0 < self.dist_map).view(b, h, w, 1)
        region2 = (self.dist_map < 0).view(b, h, w, 1)

        return torch.cat((region1, region2), dim=3)

    def generate_line_mask(self):
        b, h, w = self.dist_map.shape
        line_mask = ((-1 <= self.dist_map) * (self.dist_map <= 1)).view(b, h, w, 1)

        return line_mask

    def generate_gaussian_weight_map(self):
        b, h, w = self.dist_map.shape
        dist_map = torch.abs(self.dist_map)
        weighted_map = torch.exp(-1 * torch.pow(dist_map, 2) / (2 * self.cfg.adj_gaussian_sigma))
        weighted_map *= (torch.abs(self.dist_map) < self.cfg.region_dist)
        return weighted_map.view(b, h, w, 1)


    def update(self, line_pts, scale_factor):

        self.line_pts = line_pts / (self.size - 1) * (self.size // scale_factor - 1)
        self.sf = scale_factor

    def run(self, idx):

        output = {'region_mask': {},
                  'line_mask': {},
                  'grid': {},
                  'weight': {},
                  'dist_map': {}}

        line_pts = self.candidates[idx, :]

        for sf in self.scale_factor:
            self.update(line_pts, sf)
            line_eq, check = self.line_equation()
            self.calculate_distance(line_eq, check)
            line_mask = self.generate_line_mask()

            if self.cfg.disp_region_mask == True:

                if line_mask.shape[0] != 0:
                    self.visualize_separate_region(mask=line_mask[0],
                                                   line_pts=line_pts[0],
                                                   dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                                   file_name='line.jpg')
            output['line_mask'][sf] = line_mask

        return output


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

    def visualize_piece(self, mask, line_pts, dir_name, file_name):

        h, w, b = mask.shape
        disp_mask = np.zeros((h, w, 3))
        for i in range(b):
            temp_mask = to_3D_np(to_np(mask[:, :, i])) * self.color_list[:, i:i+1, :]

            disp_mask += temp_mask

        disp_mask = cv2.resize(np.uint8(disp_mask), (self.width, self.height))

        for i in range(line_pts.shape[0]):
            pt_1 = (line_pts[i, 0], line_pts[i, 1])
            pt_2 = (line_pts[i, 2], line_pts[i, 3])

            disp_mask = cv2.line(disp_mask, pt_1, pt_2, (255, 255, 255), 2)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp_mask)



def point_flip(l_eq, check, pts, x0, y0):
    a = l_eq[:, :, :, 0]
    b = l_eq[:, :, :, 1]
    c = l_eq[:, :, :, 2]

    x1 = x0 - 2 * a * (a * x0 + b * y0 + c) / (a * a + b * b)
    y1 = y0 - 2 * b * (a * x0 + b * y0 + c) / (a * a + b * b)

    # inf check
    d = x0[check == 1] - pts[check == 1, 0].view(-1, 1, 1)
    x1[check == 1] = x0[check == 1] - d * 2
    y1[check == 1] = y0[check == 1]
    return x1, y1
