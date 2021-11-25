from datasets.dataset_sel import *
from datasets.dataset_sl5k import *

from visualizes.visualize import *
from tests.forward import *

from evaluation.eval_func import *
from evaluation.eval_hiou import *
from evaluation.eval_line_detection import *

from post_processes.post_process import *
from libs.utils import _init_fn
from libs.load_model import *

def prepare_dataloader(cfg, dict_DB):
    # train dataloader
    if 'SEL' in cfg.dataset_name:
        dataset = Train_Dataset_SEL(cfg=cfg, datalist='train')
    elif 'SL5K' in cfg.dataset_name:
        dataset = Train_Dataset_SL5K(cfg=cfg, datalist='train')

    trainloader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.batch_size['img'],
                                              shuffle=True,
                                              num_workers=cfg.num_workers,
                                              worker_init_fn=_init_fn)

    dict_DB['trainloader'] = trainloader

    # test dataloader
    if cfg.dataset_name == 'SEL':
        dataset = Test_Dataset_SEL(cfg=cfg, datalist='test')
        testloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=cfg.batch_size['img'],
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 pin_memory=False)

    elif cfg.dataset_name == 'SEL_Hard':
        dataset = Test_Dataset_SEL_Hard(cfg=cfg, datalist='test')
        testloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=cfg.batch_size['img'],
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 pin_memory=False)

    elif 'SL5K' in cfg.dataset_name:
        dataset = Test_Dataset_SL5K(cfg=cfg, datalist='val')
        testloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=cfg.batch_size['img'],
                                                 shuffle=False,
                                                 num_workers=cfg.num_workers,
                                                 pin_memory=False)

    dict_DB['testloader'] = testloader


    return dict_DB

def prepare_model(cfg, dict_DB):

    if 'test' in cfg.run_mode:
        dict_DB = load_model_for_test(cfg, dict_DB)

    if 'train' in cfg.run_mode:
        dict_DB = load_model_for_train(cfg, dict_DB)

    dict_DB['forward_model'] = Forward_Model(cfg=cfg)

    return dict_DB


def prepare_visualization(cfg, dict_DB):

    dict_DB['visualize'] = Visualize_cv(cfg=cfg)
    return dict_DB

def prepare_evaluation(cfg, dict_DB):
    dict_DB['eval_func'] = Evaluation_Function(cfg=cfg)
    dict_DB['eval_hiou'] = Evaluation_HIoU(cfg=cfg)
    dict_DB['eval_line'] = Evaluation_Semantic_Line(cfg, dict_DB['eval_func'], dict_DB['eval_hiou'])

    return dict_DB

def prepare_post_processing(cfg, dict_DB):
    dict_DB['post_process'] = Post_Process_hough_to_line(cfg=cfg)

    return dict_DB


def prepare_training(cfg, dict_DB):

    logfile = cfg.dir['out'] + 'train/log/logfile.txt'
    mkdir(path=cfg.dir['out'] + 'train/log/')

    if cfg.run_mode == 'train' and cfg.resume == True:
        rmfile(path=logfile)
        val_result = {'AUC_P_upper_R_0.94': 0,
                      'AUC_P_upper_R_0.96': 0}



        dict_DB['val_result'] = val_result
        dict_DB['epoch'] = 0

        record_config(cfg, logfile)

    dict_DB['logfile'] = logfile

    return dict_DB

