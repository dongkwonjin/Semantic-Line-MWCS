import torch

import numpy as np

def to_tensor(data):
    return torch.from_numpy(data).cuda()

def to_np(data):
    return data.cpu().numpy()


def find_endpoints(data, size):
    x1, y1, x2, y2 = data[0], data[1], data[2], data[3]

    pts = []
    if x1 - x2 != 0:
        a = (y1 - y2) / (x1 - x2)

        b = -1
        c = -1 * a * x1 + y1

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

        # y = 0

        if y1 != y2:

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
    else:
        if x1 >= 0 and x1 <= size[0]:
            pts.append(x1)
            pts.append(0)
            pts.append(x1)
            pts.append(size[1])

    return np.float32(pts)
