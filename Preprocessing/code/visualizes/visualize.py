import cv2

import numpy as np

from libs.utils import *

class Visualize(object):

    def __init__(self, cfg):

        self.cfg = cfg

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

    # def overlap_2_on_1(self, ref_name1, ref_name2, name='overlap'):
    #     img1 = np.copy(self.show[ref_name1])
    #     img2 = self.show[ref_name2]
    #
    #     img1[img2 == 255] = 255
    #
    #     self.show[name] = img1


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

    def draw_lines_cv(self, data, name, ref_name='img', color=(255, 0, 0), s=2):
        img = np.ascontiguousarray(np.copy(self.show[ref_name]))
        for i in range(data.shape[0]):
            pts = data[i]
            pt_1 = (pts[0], pts[1])
            pt_2 = (pts[2], pts[3])
            img = cv2.line(img, pt_1, pt_2, color, s)

        self.show[name] = img

    def display_saveimg(self, dir_name, list):
        disp = self.line
        for i in range(len(list)):
            disp = np.concatenate((disp, self.show[list[i]], self.line), axis=1)

        mkdir(dir_name + '/'.join(self.show['img_name'].split('/')[:-1]))
        cv2.imwrite(dir_name + self.show['img_name'], disp)


    def display_gtlines(self, data, mode='pri'):


        self.draw_lines_cv(data=data, name='gtlines')

        self.show['overlap'] = np.copy(self.show['label'])
        if mode == 'mul' and self.cfg.datalist_mode != 'test':
            self.draw_lines_cv(data=data, name='overlap', ref_name='overlap')
        if mode == 'pri':
            self.draw_lines_cv(data=data, name='overlap', ref_name='overlap',
                               color=(0, 255, 0))

        self.display_saveimg(dir_name=self.cfg.dir['out'] + 'display/',
                             list=['img', 'gtlines', 'overlap'])

