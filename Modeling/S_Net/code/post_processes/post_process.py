import math
import numpy as np

from libs.modules import *
from libs.utils import *

class Post_Process_hough_to_line(object):

    def __init__(self, cfg=None):
        self.cfg = cfg

        self.height = cfg.height
        self.width = cfg.width

        self.center = np.array(self.cfg.center_pt)
        self.max_dist = self.cfg.max_dist

        self.candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.cand_angle = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'angle'))
        self.cand_dist = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'dist'))

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

    def run_for_cls(self, out):

        line_pts = self.candidates[to_np(out['center_idx'])]
        return line_pts

    def run_for_reg(self, out, out_cls):

        # out hough line
        idx = out_cls['center_idx']
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
        line_pts = np.float32(line_pts)
        return line_pts

    def update_data(self, batch):

        self.batch = batch


