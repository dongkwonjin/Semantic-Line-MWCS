import cv2

import numpy as np

def line_mask(line_pts, size):
    # pts - [n, 4]
    num = line_pts.shape[0]

    data = []
    for i in range(num):

        pt_1 = line_pts[i, 0], line_pts[i, 1]
        pt_2 = line_pts[i, 2], line_pts[i, 3]

        line_mask = np.zeros((size[0], size[1]), np.int32)
        line_mask = cv2.line(line_mask, pt_1, pt_2, 1, 2)

        data.append(line_mask)

    data = np.array(data)
    return data

