import os
import torch

class Config(object):
    def __init__(self):
        self.settings_for_system()
        self.settings_for_path()

        self.settings_for_image_processing()
        self.settings_for_dataloading()
        self.settings_for_training()
        self.settings_for_evaluation()
        self.settings_for_hyperparam()
        self.settings_for_thresd()

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
        self.dir['head'] = '--system root'  # need to modify
        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        self.dir['preprocess'] = dict()
        self.dir['preprocess']['SEL'] = self.dir['head'] + '--preprocessed_data root /SEL/pickle/'  # need to modify
        self.dir['preprocess']['SEL_Hard'] = '--preprocessed_data root /SEL_Hard/pickle/'  # need to modify
        self.dir['preprocess']['SL5K'] = self.dir['head'] + '--preprocessed_data root /SL5K/pickle/'  # need to modify

        self.settings_dataset_path()

        self.dir['pretrained_snet'] = self.dir['head'] + '--Pretrained snet PATH'  # need to modify

        self.dir['out'] = self.dir['proj'] + 'output_hnet_{}/'.format(self.dataset_name)
        self.dir['weight'] = self.dir['out'] + 'train/weight/'
        self.dir['paper_weight'] = self.dir['head'] + '--paper_weight root'  # need to modify

    def settings_dataset_path(self):
        self.dataset_name = 'SL5K'  # ['SEL', 'SEL_Hard', 'SL5K']  # SL5K --> Nankai

        self.dir['dataset'] = dict()
        self.dir['dataset']['SEL'] = self.dir['head'] + '--dataset root /SEL/'
        self.dir['dataset']['SEL_Hard'] = self.dir['head'] + '--dataset root /SEL_Hard/'
        self.dir['dataset']['SL5K'] = self.dir['head'] + '--dataset root /SL5K/'
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

    def settings_for_training(self):
        self.run_mode = 'test_paper'  # ['train', 'test', 'eval', 'test_paper']
        self.resume = False

        self.epochs = 500
        self.batch_size = {'img': 1}
        self.max_iter = 8
        self.max_iter_train = 10
        self.max_num = 10
        self.case_num = 7

        # optimizer
        self.lr = 1e-5
        self.milestones = [60, 120, 180, 240]
        self.weight_decay = 5e-4
        self.gamma = 0.5
        self.loss_fn = 'MSE'  # ['BCE', 'MSE']

        self.use_pretrained_snet = False

    def settings_for_dataloading(self):
        self.num_workers = 4

        self.gaussian_blur_seg = True
        self.kernel = (9, 9)
        self.downscale = 'bilinear'
        self.ratio = 16
        self.crop_size = 0

    def settings_for_evaluation(self):
        self.epoch_eval = -1
        self.eval_semantic_line = True

        self.sf_hiou = 1


    def settings_for_visualization(self):
        self.disp_test_result = True
        self.disp_step = 50
        self.disp_region_mask = False
        self.disp_post_process = False
        self.disp_sim_map_edge = False
        self.disp_sim_map_vertex = False
        self.disp_edge_score = False
        self.disp_train_per_case = True

        self.disp_hiou = False

    def settings_for_save(self):
        self.save_pickle = True

    def settings_for_hyperparam(self):
        self.cluster_num = 9
        self.max_offset_a = 10
        self.max_offset_d = 20
        self.region_dist = 10  # ['3', 'self.max_dist']
        self.adj_gaussian_sigma = 10.0
        self.scale_factor = [8, 16]

    def settings_for_thresd(self):
        self.constrain_max_score = False
