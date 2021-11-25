import torch
from networks.model_snet import Model as S_Net
from networks.model_hnet import Model as H_Net
from networks.loss import *

def load_model_for_test(cfg, dict_DB):
    dict_DB = load_model_snet(cfg, dict_DB)

    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.dir['paper_weight'] + 'checkpoint_paper_H_Net_{}'.format(cfg.dataset_name.replace('_Hard', '')))
    elif cfg.run_mode == 'test':
        # checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final')
        checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_AUC_F_{}'.format(cfg.dataset_name.replace('_Hard', '')))
    model = H_Net(cfg=cfg)

    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['H_Net'] = model

    return dict_DB

def load_model_for_train(cfg, dict_DB):
    dict_DB = load_model_snet(cfg, dict_DB)

    model = H_Net(cfg=cfg)
    model.cuda()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.milestones,
                                                     gamma=cfg.gamma)

    if cfg.resume == False:
        checkpoint = torch.load(cfg.dir['weight'] + 'checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.milestones,
                                                         gamma=cfg.gamma,
                                                         last_epoch=checkpoint['epoch'])
        dict_DB['epoch'] = checkpoint['epoch'] + 1
        dict_DB['val_result'] = checkpoint['val_result']

    if cfg.loss_fn == 'MSE':
        loss_fn = Loss_Function_MSE()

    dict_DB['H_Net'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB


def load_model_snet(cfg, dict_DB):
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(cfg.dir['paper_weight'] + 'checkpoint_paper_S_Net_{}'.format(cfg.dataset_name.replace('_Hard', '')))
    else:
        checkpoint = torch.load(cfg.dir['pretrained_snet'] + 'checkpoint_EA_P_upper_R_0.86_'.format(cfg.dataset_name.replace('_Hard', '')))
    model = S_Net(cfg=cfg)

    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['S_Net'] = model

    return dict_DB