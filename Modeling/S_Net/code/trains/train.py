import torch

from libs.save_model import *
from libs.utils import *

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']
        self.s_net = dict_DB['S_Net']
        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']

        self.test_process = dict_DB['test_process']
        self.eval_func = dict_DB['eval_func']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.batch_size = cfg.batch_size['img']


    def training(self):
        loss_t = {'sum': 0, 'prob': 0, 'off_a': 0, 'off_d': 0}

        # train start
        self.s_net.train()
        print('train start')
        logger('train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):

            # load data
            img = batch['img'].cuda()
            prob = batch['prob'].cuda()
            offset = batch['offset'].cuda()
            img_name = batch['img_name'][0]

            # model
            out_cls = self.s_net.forward_for_cls(img, is_training=True)
            out_reg = self.s_net.forward_for_reg(l_feat=out_cls['l_feat'],
                                                 idx=torch.arange(out_cls['l_feat'].shape[2]).cuda())
            out = dict(out_cls, **out_reg)
            # loss
            loss = self.loss_fn(
                out=out,
                gt_prob=prob,
                gt_offset=offset)

            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()


            loss_t['sum'] += loss['sum'].item()
            loss_t['prob'] += loss['prob'].item()
            loss_t['off_a'] += loss['off_a'].item()
            loss_t['off_d'] += loss['off_d'].item()

            # display
            if i % self.cfg.disp_step == 0:
                print('img iter %d ==> %s' % (i, img_name))
                self.visualize.display_for_train_reg(batch, out, i)
                if i % self.cfg.disp_step == 0:
                    logger("%d %s ==> Loss : %5f, Loss_prob : %5f, Loss_off_a : %5f, Loss_off_d : %5f\n"
                           % (i, img_name, loss['sum'].item(), loss['prob'].item(), loss['off_a'].item(), loss['off_d'].item()), self.logfile)

        # logger
        logger("Average Loss : %5f %5f %5f %5f\n"
               % (loss_t['sum'] / i, loss_t['prob'] / i, loss_t['off_a'] / i, loss_t['off_d'] / i), self.logfile)

        print("Average Loss : %5f %5f %5f %5f\n"
               % (loss_t['sum'] / i, loss_t['prob'] / i, loss_t['off_a'] / i, loss_t['off_d'] / i))

        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.s_net,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}

        save_model(checkpoint=self.ckpt,
                   param='checkpoint_final',
                   path=self.cfg.dir['weight'])

    def validation(self):
        metric = self.test_process.run(self.s_net, mode='val', dataset_name=self.cfg.dataset_name)

        logger("Epoch %03d => SEL EA_P_reg %5f / EA_R_reg %5f / EA_F_reg %5f \n"
               % (self.ckpt['epoch'], metric['ea_p'], metric['ea_r'], metric['ea_f']), self.logfile)
        self.val_result['EA_P_upper_R_0.84'] = save_model_max_upper(self.ckpt, self.cfg.dir['weight'],
                                                                     self.val_result['EA_P_upper_R_0.84'], metric['ea_p'], metric['ea_r'], 0.84,
                                                                     logger, self.logfile, 'EA_P_upper_R_0.84', dataset_name=self.cfg.dataset_name)
        self.val_result['EA_P_upper_R_0.86'] = save_model_max_upper(self.ckpt, self.cfg.dir['weight'],
                                                                     self.val_result['EA_P_upper_R_0.86'], metric['ea_p'], metric['ea_r'], 0.86,
                                                                     logger, self.logfile, 'EA_P_upper_R_0.86', dataset_name=self.cfg.dataset_name)

        logger("Epoch %03d => SEL AUC_A %5f / AUC_P %5f / AUC_R %5f\n"
               % (self.ckpt['epoch'], metric['auc_a'], metric['auc_p'], metric['auc_r']), self.logfile)
        self.val_result['AUC_P_upper_R_0.94'] = save_model_max_upper(self.ckpt, self.cfg.dir['weight'],
                                                                     self.val_result['AUC_P_upper_R_0.94'], metric['auc_p'], metric['auc_r'], 0.94,
                                                                     logger, self.logfile, 'AUC_P_upper_R_0.94', dataset_name=self.cfg.dataset_name)
        self.val_result['AUC_P_upper_R_0.96'] = save_model_max_upper(self.ckpt, self.cfg.dir['weight'],
                                                                     self.val_result['AUC_P_upper_R_0.96'], metric['auc_p'], metric['auc_r'], 0.96,
                                                                     logger, self.logfile, 'AUC_P_upper_R_0.96', dataset_name=self.cfg.dataset_name)

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch
            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)

            self.training()
            if epoch > self.cfg.epoch_eval:
                self.validation()
            self.scheduler.step(self.epoch)

