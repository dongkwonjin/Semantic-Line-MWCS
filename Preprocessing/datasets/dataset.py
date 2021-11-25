import cv2

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from libs.utils import *


class Dataset_train(Dataset):
    def __init__(self, cfg, datalist):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.datalist = load_pickle(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name], 'data/', datalist))
        try:
            self.imglist = self.datalist['img_path']
        except:
            self.imglist = self.datalist['img_name']
        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.size = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx):
        if self.cfg.dataset_name == 'SEL':
            img = Image.open(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name + '_img'], self.imglist[idx])).convert('RGB')
        elif self.cfg.dataset_name == 'SL5K':
            img = Image.open(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name + '_img'], 'train', self.imglist[idx])).convert('RGB')

        size = [img.size[1], img.size[0]]
        img = img.crop((0, self.cfg.crop_size, int(size[1]), int(size[0])))

        # transform
        img = self.transform(img)

        return img

    def get_label(self, idx):
        label = {'mul_gt': []}
        label['mul_gt'] = self.datalist['multiple'][idx]
        return label

    def __getitem__(self, idx):
        img = self.get_image(idx)
        label = self.get_label(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'label': label,
                'img_name': self.imglist[idx]}

    def __len__(self):
        return len(self.imglist)


class Dataset_test(Dataset):
    def __init__(self, cfg, datalist='test'):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width

        self.datalist = load_pickle(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name], 'data/', datalist))
        try:
            self.imglist = self.datalist['img_path']
        except:
            self.imglist = self.datalist['img_name']
        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width),
                                                               interpolation=2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        self.size = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

    def transform(self, img, interpolation):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (self.width, self.height), interpolation=interpolation)
        return img

    def get_image(self, idx):
        if self.cfg.dataset_name == 'SEL':
            img = Image.open(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name + '_img'], self.imglist[idx])).convert('RGB')
        elif self.cfg.dataset_name == 'SL5K':
            img = Image.open(os.path.join(self.cfg.dir['dataset'][self.cfg.dataset_name + '_img'], 'val', self.imglist[idx])).convert('RGB')

        size = [img.size[1], img.size[0]]
        img = img.crop((0, self.cfg.crop_size, int(size[1]), int(size[0])))

        # transform
        img = self.transform(img)

        return img

    def get_label(self, idx):
        label = {'pri_gt': [], 'mul_gt': []}
        label['mul_gt'] = self.datalist['multiple'][idx]
        return label

    def __getitem__(self, idx):
        img = self.get_image(idx)
        label = self.get_label(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'label': label,
                'img_name': self.imglist[idx]}

    def __len__(self):
        return len(self.imglist)