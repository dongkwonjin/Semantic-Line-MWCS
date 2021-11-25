import os
import torch

class Config(object):
    def __init__(self):
        self.settings_for_system()
        self.settings_for_path()

        self.settings_for_image_processing()
        self.settings_for_preprocessing()
        self.settings_for_dataloading()

        self.settings_for_visualization()
        self.settings_for_save()

    def settings_for_system(self):
        self.gpu_id = "0"
        self.seed = 123

        # GPU setting
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
        torch.backends.cudnn.deterministic = True

    def settings_for_path(self):
        self.dir = dict()
        self.dir['head'] = '/media/dkjin/4fefb28c-5de9-4abd-a935-aa2d61392048/'
        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'

        self.settings_dataset_path()

        self.dir['out'] = self.dir['proj'] + 'output_{}_{}/'.format(self.dataset_name, self.datalist_mode)

    def settings_dataset_path(self):
        self.dataset_name = 'SL5K'  # ['SEL', 'SL5K']  # SL5K --> Nankai
        self.datalist_mode = 'train'  # ['train', 'test', 'val']

        self.dir['dataset'] = dict()
        self.dir['dataset']['SEL'] = self.dir['head'] + 'Github/Semantic-Line-DRM/Dataset/SEL/'
        self.dir['dataset']['SEL_Hard'] = self.dir['head'] + 'Github/Semantic-Line-DRM/Dataset/SEL_Hard/'
        self.dir['dataset']['SL5K'] = self.dir['head'] + 'Work/CVPR2021/Semantic_line_detection/Project_final/P04_Dataset/sl5k/'
        self.dir['dataset']['SEL_img'] = self.dir['dataset']['SEL'] + 'ICCV2017_JTLEE_images/'
        self.dir['dataset']['SEL_Hard_img'] = self.dir['dataset']['SEL_Hard'] + 'images/'
        self.dir['dataset']['SL5K_img'] = self.dir['dataset']['SL5K']

    def settings_for_image_processing(self):
        self.org_height = 400
        self.org_width = 400
        self.height = 400
        self.width = 400
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.center_pt = [(self.width - 1) / 2, (self.height - 1) / 2]
        self.max_dist = self.width // 2 - 20

    def settings_for_dataloading(self):
        self.num_workers = 4
        self.batch_size = {'img': 1}
        self.data_flip = True
        self.crop_size = 0

    def settings_for_visualization(self):
        self.display = True
        self.display_offset_validation = True

    def settings_for_save(self):
        self.save_pickle = True

    def settings_for_preprocessing(self):
        self.scale_factor = 4
        self.sampling_mode = 'grid'  # ['threshold', 'grid']
        self.min_angle_error = 0.08
        self.min_dist_error = 0.08
        self.max_offset = 80