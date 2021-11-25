import math
import cv2
import torch

import numpy as np

def generate_line_mask(line_pts, size):
    # pts - [n, 4]
    num = line_pts.shape[0]
    data = []
    for i in range(num):


        pt_1 = line_pts[i, 0], line_pts[i, 1]
        pt_2 = line_pts[i, 2], line_pts[i, 3]

        line_mask = np.zeros((size[0], size[1]), np.float32)
        line_mask = cv2.line(line_mask, pt_1, pt_2, 1, 2)

        data.append(line_mask)

    data = np.array(data)
    return data

def calculate_edge_score(ref_mask, line_mask):

    score = ref_mask * line_mask
    score = torch.sum(score, dim=(1, 2)) / (torch.sum(line_mask, dim=(1, 2)) + 1e-7)

    return score

def candidate_line_filtering(pts, size, thresd_boundary, thresd_length):
    # exclude outlier line: 1. short length
    #                       2. distance from image boundary

    check = 0

    if pts.shape[0] == 0:
        check = 1
        return check

    pt_1 = pts[:2]
    pt_2 = pts[2:]

    length = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

    # short length
    if length < thresd_length:
        check += 1

    # boundary
    if (pt_1[0] < thresd_boundary) and (pt_2[0] < thresd_boundary):
        check += 1
    if (pt_1[1] < thresd_boundary) and (pt_2[1] < thresd_boundary):
        check += 1
    if (abs(pt_1[0] - size[1]) < thresd_boundary) and (abs(pt_2[0] - size[1]) < thresd_boundary):
        check += 1
    if (abs(pt_1[1] - size[0]) < thresd_boundary) and (abs(pt_2[1] - size[0]) < thresd_boundary):
        check += 1

    return check



# line parameters

def line_equation(data):

    # data: [N, 4] numpy array  x1, y1, x2, y2 (W, H, W, H)
    line_eq = torch.zeros((data.shape[0], 3)).cuda()
    line_eq[:, 0] = (data[:, 1] - data[:, 3]) / (data[:, 0] - data[:, 2])
    line_eq[:, 1] = -1
    line_eq[:, 2] = (-1 * line_eq[:, 0] * data[:, 0]) + data[:, 1]
    check = ((data[:, 0] - data[:, 2]) == 0)

    return line_eq, check

def transform_theta_to_angle(line_eq):
    line_angle = line_eq[:, 0].clone()
    line_angle = torch.atan(line_angle) * 180 / math.pi
    return line_angle


def calculate_distance_from_center(line_eq, check, line_pts, center_pts):  # line-point distance

    num = line_eq.shape[0]
    a = line_eq[:, 0].view(num, 1, 1)
    b = line_eq[:, 1].view(num, 1, 1)
    c = line_eq[:, 2].view(num, 1, 1)

    dist = (center_pts[0] * a + center_pts[1] * b + c) / torch.sqrt(a * a + b * b)

    if True in check:
        dist[check == True] = (center_pts[0] - line_pts[check == True, 0]).view(-1, 1, 1)

    return dist

def find_endpoints_from_line_eq(line_eq, size):
    a, b, c = line_eq

    pts = []
    if a == 1 and b == 0:
        x1 = c
        if x1 >= 0 and x1 <= size[0]:
            pts.append(x1)
            pts.append(0)
            pts.append(x1)
            pts.append(size[1])

    else:
        # x = 0
        cx = 0
        cy = a * 0 + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)
        # x = size[0]
        cx = size[0]
        cy = a * size[0] + c
        if cy >= 0 and cy <= size[1]:
            pts.append(cx)
            pts.append(cy)

        if a != 0:
            # y = 0
            cx = (0 - c) / a
            cy = 0
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)

            # y = size[1]
            cx = (size[1] - c) / a
            cy = size[1]
            if cx >= 0 and cx <= size[0]:
                pts.append(cx)
                pts.append(cy)
    pts = np.float32(pts)
    if pts.shape[0] != 0:
        pts = np.unique(pts.reshape(-1, 2), axis=0).reshape(-1)
    return pts
