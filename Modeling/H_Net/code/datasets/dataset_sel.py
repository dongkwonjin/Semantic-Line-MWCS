import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.utils import *


class Train_Dataset_SEL(Dataset):
    def __init__(self, cfg, datalist='train'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        datalist = load_pickle(os.path.join(self.cfg.dir['dataset']['SEL'], 'data/', datalist))
        self.img_list = datalist['img_path']

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.size = np.float32(self.cfg.size)

        candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.c_num = candidates.shape[0]

        # hough space
        self.max_dist = self.cfg.max_dist
        hough_space = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'hough_space')
        self.hough_space = hough_space
        self.hough_h, self.hough_w = self.hough_space['idx'].shape
        self.hough_space_idx = self.hough_space['idx'][self.hough_h // 3:self.hough_h // 3 * 2]
        self.outlier_mask = (self.hough_space_idx != -1)
        self.angle_list = np.float32(self.hough_space['angle_list']) / 90
        self.dist_list = np.float32(self.hough_space['dist_list']) / self.max_dist

        # gaussian weight
        self.sigma_a = 1.0
        self.sigma_d = 1.5

        self.mean_d = 0

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx, flip=0):
        img = Image.open(os.path.join(self.cfg.dir['dataset']['SEL_img'], self.img_list[idx])).convert('RGB')

        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        size = [img.size[1], img.size[0]]
        img = img.crop((0, self.cfg.crop_size, int(size[1]), int(size[0])))

        # transform
        img = self.transform(img)

        return img

    def get_edge_map(self, line_pts, scale_factor=1):
        line_pts = np.concatenate(line_pts)
        edge_map = np.zeros((self.height // scale_factor, self.width // scale_factor), dtype=np.uint8)

        for i in range(line_pts.shape[0]):
            endpts = line_pts[i] / (self.size - 1) * (self.size // scale_factor - 1)
            pt_1 = (endpts[0], endpts[1])
            pt_2 = (endpts[2], endpts[3])
            edge_map = cv2.line(edge_map, pt_1, pt_2, 255, 2)

        return np.float32(edge_map / 255)

    def get_transformed_label(self, idx, flip):
        # load data
        data = load_pickle(os.path.join(self.cfg.dir['preprocess'][self.cfg.dataset_name], self.img_list[idx][:-4]))
        mul_data = data[flip]['mul']

        # gtlines
        gtlines = np.concatenate(mul_data['pts'])

        # mul line
        line_num = len(mul_data['region_idx'])

        # idx
        a_idx = np.array(mul_data['angle_idx'])
        d_idx = np.array(mul_data['dist_idx'])

        return gtlines, a_idx, d_idx

    def __getitem__(self, idx):
        flip = random.randint(0, 1)

        img = self.get_image(idx, flip)

        gtlines, a_idx, d_idx = self.get_transformed_label(idx, flip)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'gtlines': gtlines,
                'a_idx': a_idx,
                'd_idx': d_idx,
                'pri_gt': gtlines,
                'mul_gt': gtlines,
                'img_name': self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)

class Test_Dataset_SEL(Dataset):
    def __init__(self, cfg, datalist='test'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        # load datalist
        self.datalist = load_pickle(os.path.join(self.cfg.dir['dataset']['SEL'], 'data/', datalist))
        self.img_list = self.datalist['img_path']

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.size = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

        candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.c_num = candidates.shape[0]

        # gaussian weight
        self.sigma_a = 1.0
        self.sigma_d = 1.5

        self.mean_d = 0

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx, flip=0):
        img = Image.open(os.path.join(self.cfg.dir['dataset']['SEL_img'], self.img_list[idx])).convert('RGB')

        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        size = [img.size[1], img.size[0]]
        img = img.crop((0, self.cfg.crop_size, int(size[1]), int(size[0])))

        # transform
        img = self.transform(img)

        return img

    def get_gtlines(self, idx):

        mul_gt = self.datalist['multiple'][idx]
        pri_gt_idx = self.datalist['primary'][idx]
        pri_gt = mul_gt[pri_gt_idx == 1]

        return pri_gt, mul_gt

    def __getitem__(self, idx):

        img = self.get_image(idx)
        pri_gt, mul_gt = self.get_gtlines(idx)
        return {'img_rgb': img,
                'img': self.normalize(img),
                'pri_gt': pri_gt,
                'mul_gt': mul_gt,
                'img_name': self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)

class Test_Dataset_SEL_Hard(Dataset):
    def __init__(self, cfg, datalist='test'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        # load datalist
        self.datalist = load_pickle(os.path.join(self.cfg.dir['dataset']['SEL_Hard'], 'data/', datalist))
        self.img_list = self.datalist['img_path']

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.size = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

        candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')
        self.c_num = candidates.shape[0]

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx, flip=0):
        img = Image.open(os.path.join(self.cfg.dir['dataset']['SEL_Hard_img'], self.img_list[idx])).convert('RGB')

        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        size = [img.size[1], img.size[0]]
        img = img.crop((0, self.cfg.crop_size, int(size[1]), int(size[0])))

        # transform
        img = self.transform(img)

        return img

    def get_gtlines(self, idx):

        mul_gt = self.datalist['multiple'][idx]
        pri_gt = self.datalist['primary'][idx]

        return pri_gt, mul_gt

    def __getitem__(self, idx):

        img = self.get_image(idx)
        pri_gt, mul_gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'pri_gt': pri_gt,
                'mul_gt': mul_gt,
                'img_name': self.img_list[idx]}

    def __len__(self):
        return len(self.img_list)
