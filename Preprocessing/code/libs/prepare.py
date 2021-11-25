from datasets.dataset import *
from visualizes.visualize import *

def prepare_dataloader(cfg, dict_DB):

    if 'train' in cfg.datalist_mode:
        dataset = Dataset_train(cfg=cfg, datalist=cfg.datalist_mode)
    else:
        dataset = Dataset_test(cfg=cfg, datalist=cfg.datalist_mode)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=cfg.batch_size['img'],
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=False)

    dict_DB['dataloader'] = dataloader

    return dict_DB

def prepare_visualization(cfg, dict_DB):

    dict_DB['visualize'] = Visualize(cfg)
    return dict_DB

