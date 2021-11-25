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

        # region1 = ((0 < self.dist_map) * (self.dist_map < self.cfg.region_dist)).view(b, h, w, 1)
        # region2 = ((-self.cfg.region_dist < self.dist_map) * (self.dist_map < 0)).view(b, h, w, 1)

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
            region_mask = self.generate_region_mask()
            line_mask = self.generate_line_mask()
            weight = self.generate_gaussian_weight_map()
            # grid = self.generate_flipped_grid()
            # dist_map = self.dist_map

            if self.cfg.disp_region_mask == True:

                if region_mask.shape[0] != 0:
                    self.visualize_separate_region(mask=region_mask[0],
                                                   line_pts=line_pts[0],
                                                   dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                                   file_name='region.jpg')
            output['region_mask'][sf] = region_mask
            output['line_mask'][sf] = line_mask
            output['weight'][sf] = weight
            # output['grid'][sf] = grid
            # output['dist_map'][sf] = dist_map

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


class Inter_Region(nn.Module):
    def __init__(self, cfg):
        super(Inter_Region, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.candidates = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates'))

        self.color_list = np.array([[[0, 0, 255],
                                    [0, 0, 128],
                                    [0, 255, 0],
                                    [0, 128, 0]]])

    def get_inter_mask(self, r_mask1, r_mask2):
        b, _, h, w = r_mask1.shape

        out1 = r_mask1 * r_mask2
        out2 = r_mask1 * r_mask2[:, [1, 0]]
        inter_mask = torch.cat((out1, out2), dim=1)

        if self.cfg.disp_region_mask == True:
            for k in range(b):
                temp_line_pts = torch.cat((self.line_pts1[k:k + 1], self.line_pts2[k:k + 1]), dim=0)
                self.visualize_inter_region_all(mask=inter_mask[k],
                                                line_pts=temp_line_pts,
                                                dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                                file_name='inter_region_all_' + str(k) + '.jpg')

                # break

        return inter_mask

    def forward_adj_inter_mask(self, inter_mask):
        idx = [[0, 2], [1, 3], [0, 3], [1, 2]]
        b, _, h, w = inter_mask.shape
        adj_inter_mask = inter_mask[:, idx, :, :]

        n_mask = (torch.sum(adj_inter_mask, dim=(3, 4), keepdim=True) != 0)
        check = torch.mul(n_mask[:, :, 0:1], n_mask[:, :, 1:2])


        if self.cfg.disp_region_mask == True:
            for k in range(b):
                temp_line_pts = torch.cat((self.line_pts1[k:k + 1], self.line_pts2[k:k + 1]), dim=0)

                self.visualize_adj_region_all(mask=adj_inter_mask[k] * check[k],
                                              line_pts=temp_line_pts,
                                              dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                              file_name='adj_region_' + str(k) + '.jpg')
                # break


        return adj_inter_mask * check



    def update_for_visualize(self, img, pair_idx):
        img = to_np(img[0].permute(1, 2, 0))
        self.img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.line_pts1 = self.candidates[pair_idx[:, 0]]
        self.line_pts2 = self.candidates[pair_idx[:, 1]]


    def visualize(self, mask, line_pts, dir_name, file_name):
        mask = self.Colormap(self.img, np.uint8(to_np(mask)))
        mask = np.ascontiguousarray(np.uint8(mask))
        for i in range(line_pts.shape[0]):
            pt_1 = (line_pts[i, 0], line_pts[i, 1])
            pt_2 = (line_pts[i, 2], line_pts[i, 3])

            mask = cv2.line(mask, pt_1, pt_2, (0, 0, 255), 3)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, mask)

    def visualize_inter_region_all(self, mask, line_pts, dir_name, file_name):

        b, h, w = mask.shape
        disp_mask = np.zeros((h, w, 3))
        for i in range(b):
            temp_mask = to_3D_np(to_np(mask[i, :, :])) * self.color_list[:, i:i+1, :]

            disp_mask += temp_mask

        disp_mask = cv2.resize(np.uint8(disp_mask), (self.width, self.height))

        for i in range(line_pts.shape[0]):
            pt_1 = (line_pts[i, 0], line_pts[i, 1])
            pt_2 = (line_pts[i, 2], line_pts[i, 3])

            if i == 0:
                disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 0, 255), 2)
            elif i == 1:
                disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 255, 0), 2)
        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, disp_mask)

    def visualize_adj_region_all(self, mask, line_pts, dir_name, file_name):

        b, _, h, w = mask.shape
        for i in range(4):
            disp_mask = np.zeros((h, w, 3))
            temp_mask = to_3D_np(to_np(mask[i, 0, :, :])) * self.color_list[:, i:i+1, :]
            disp_mask += temp_mask
            temp_mask = to_3D_np(to_np(mask[i, 1, :, :])) * self.color_list[:, i:i+1, :]
            disp_mask += temp_mask

            disp_mask = cv2.resize(np.uint8(disp_mask), (self.width, self.height))

            for j in range(line_pts.shape[0]):
                pt_1 = (line_pts[j, 0], line_pts[j, 1])
                pt_2 = (line_pts[j, 2], line_pts[j, 3])

                if j == 0:
                    disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 0, 255), 2)
                elif j == 1:
                    disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 255, 0), 2)
            mkdir(dir_name)
            cv2.imwrite(dir_name + file_name + '_' + str(i), disp_mask)


    def Colormap(self, img, mask, alpha=0.5):

        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.uint8(cv2.resize(mask, (self.width, self.height)) * 255)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        out = img * alpha + mask * (1 - alpha)

        return out


class Adjacent_Region(nn.Module):
    def __init__(self, cfg):
        super(Adjacent_Region, self).__init__()
        # cfg
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.candidates = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates'))

        self.color_list = np.array([[[0, 0, 125],
                                    [0, 125, 0],
                                    [125, 0, 0],
                                    [0, 125, 125]]])

    def overlapping_regions(self, mask1, mask2, grid):
        b, h, w, _ = grid.shape

        if len(mask1.shape) == 3:
            mask1 = mask1.view(b, 1, h, w)
            mask2 = mask2.view(b, 1, h, w)

        if mask1.dtype != torch.float32:
            mask1 = mask1.type(torch.float)
            mask2 = mask2.type(torch.float)

        # flip
        out1 = F.grid_sample(mask1, grid, align_corners=True)
        out2 = F.grid_sample(mask2, grid, align_corners=True)
        return ((out1 * mask2 + out2 * mask1) != 0)


    def get_similarity_map(self):
        b, c, h, w = self.s_vec1.shape

        s_vec = torch.cat((self.s_vec2.view(b, 1, c, h, w).repeat(1, 2, 1, 1, 1),
                           self.s_vec1.view(b, 1, c, h, w).repeat(1, 2, 1, 1, 1)), dim=1)

        sim_data = s_vec * self.overlap_mask.view(b, 4, 1, h, w)
        return sim_data

    def get_overlap_mask(self):
        b, _, h, w = self.inter_mask.shape
        self.overlap_mask = torch.BoolTensor([]).cuda()

        out1 = self.overlapping_regions(self.inter_mask[:, [0, 1]], self.inter_mask[:, [2, 3]], self.grid2)
        out2 = self.overlapping_regions(self.inter_mask[:, [0, 1]], self.inter_mask[:, [3, 2]], self.grid1)
        self.overlap_mask = torch.cat((out1, out2), dim=1)

        if self.cfg.disp_region_mask == True:
            for k in range(b):
                temp_line_pts = torch.cat((self.line_pts1[k:k + 1], self.line_pts2[k:k + 1]), dim=0)
                for i in range(4):

                    self.visualize(mask=self.overlap_mask[k, i],
                                   line_pts=temp_line_pts,
                                   dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                   file_name='overlap_region_' + str(k) + '_' + str(i) + '.jpg')


    def forward_overlap_mask(self, inter_mask, grid1, grid2, weight1=None, weight2=None):

        self.grid1 = grid1
        self.grid2 = grid2
        self.inter_mask = inter_mask

        self.get_overlap_mask()

        return self.overlap_mask

    def forward_att_mask(self, adj_inter_mask, weight1, weight2):
        b, _, h, w = weight1.shape
        att_mask1 = adj_inter_mask[:, :2] * weight2.view(b, 1, 1, h, w)
        att_mask2 = adj_inter_mask[:, 2:] * weight1.view(b, 1, 1, h, w)
        att_mask = torch.cat((att_mask1, att_mask2), dim=1)

        if self.cfg.disp_region_mask == True:
            for k in range(b):
                temp_line_pts = torch.cat((self.line_pts1[k:k + 1], self.line_pts2[k:k + 1]), dim=0)

                self.visualize_adj_region_all(mask=att_mask[k],
                                              line_pts=temp_line_pts,
                                              dir_name=self.cfg.dir['out'] + 'vis_check/region_mask/',
                                              file_name='att_adj_region_' + str(k) + '.jpg')
        return att_mask


    def forward_sim_map(self, feat, grid1, grid2):
        b, c, h, w = feat.shape
        batch_size = grid1.shape[0]
        feat = feat.expand(batch_size, c, h, w)

        self.s_vec1 = Similarity_vector(x=feat, grid=grid1)
        self.s_vec2 = Similarity_vector(x=feat, grid=grid2)

        sim_data = self.get_similarity_map()
        sim_map = torch.sum(sim_data, dim=2, keepdim=True)

        return sim_data, sim_map

    def forward_sim_map_v2(self, feat, r_mask1, r_mask2, grid1, grid2):
        _, c, h, w = feat.shape
        b, _, _, _ = r_mask1.shape
        batch_size = grid1.shape[0]
        feat = feat.expand(batch_size, c, h, w)

        self.s_vec1 = Similarity_vector(x=feat, grid=grid1).view(b, 1, c, h, w)
        self.s_vec2 = Similarity_vector(x=feat, grid=grid2).view(b, 1, c, h, w)
        s_vec = torch.cat((self.s_vec1, self.s_vec2), dim=1)
        r_mask1 = torch.sum(r_mask1.view(b, 2, 1, h, w), dim=1, keepdim=True)
        r_mask2 = torch.sum(r_mask2.view(b, 2, 1, h, w), dim=1, keepdim=True)
        r_mask = torch.cat((r_mask1, r_mask2), dim=1)
        sim_map = torch.sum(s_vec * r_mask, dim=2, keepdim=True)

        return _, sim_map

    def update_for_visualize(self, img, pair_idx):
        img = to_np(img[0].permute(1, 2, 0))
        self.img = np.uint8((img * self.std + self.mean) * 255)[:, :, [2, 1, 0]]
        self.line_pts1 = self.candidates[pair_idx[:, 0]]
        self.line_pts2 = self.candidates[pair_idx[:, 1]]


    def visualize(self, mask, line_pts, dir_name, file_name):
        mask = self.Colormap(self.img, np.uint8(to_np(mask)))
        mask = np.ascontiguousarray(np.uint8(mask))
        for i in range(line_pts.shape[0]):
            pt_1 = (line_pts[i, 0], line_pts[i, 1])
            pt_2 = (line_pts[i, 2], line_pts[i, 3])

            mask = cv2.line(mask, pt_1, pt_2, (0, 0, 255), 3)

        mkdir(dir_name)
        cv2.imwrite(dir_name + file_name, mask)

    def visualize_adj_region_all(self, mask, line_pts, dir_name, file_name):

        b, _, h, w = mask.shape
        for i in range(4):
            disp_mask = np.zeros((h, w, 3))
            temp_mask = to_3D_np(to_np(mask[i, 0, :, :])) * self.color_list[:, i:i+1, :]
            disp_mask += temp_mask
            temp_mask = to_3D_np(to_np(mask[i, 1, :, :])) * self.color_list[:, i:i+1, :]
            disp_mask += temp_mask

            disp_mask = cv2.resize(np.uint8(disp_mask), (self.width, self.height))

            for j in range(line_pts.shape[0]):
                pt_1 = (line_pts[j, 0], line_pts[j, 1])
                pt_2 = (line_pts[j, 2], line_pts[j, 3])

                if j == 0:
                    disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 0, 255), 2)
                elif j == 1:
                    disp_mask = cv2.line(disp_mask, pt_1, pt_2, (0, 255, 0), 2)
            mkdir(dir_name)
            cv2.imwrite(dir_name + file_name + '_' + str(i), disp_mask)

    def Colormap(self, img, mask, alpha=0.5):

        mask = cv2.resize(mask, (self.width, self.height))
        mask = np.uint8(cv2.resize(mask, (self.width, self.height)) * 255)
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

        out = img * alpha + mask * (1 - alpha)

        return out

def Similarity_vector(x, grid):
    # flip
    x_fold = F.grid_sample(x, grid, align_corners=True)

    # l2_normalization
    x = l2_normalization(x)
    x_fold = l2_normalization(x_fold)

    out = x * x_fold

    return out


def Similarity_map(x, grid):
    # flip
    x_fold = F.grid_sample(x, grid, align_corners=True)

    # l2_normalization
    x = l2_normalization(x)
    x_fold = l2_normalization(x_fold)

    s_map = torch.sum(x * x_fold, dim=1, keepdim=True)

    return s_map

def l2_normalization(x):
    ep = 1e-5
    out = x / (torch.norm(x, p=2, dim=1, keepdim=True) + ep)
    return out

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
