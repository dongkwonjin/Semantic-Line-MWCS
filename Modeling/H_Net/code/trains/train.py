import torch

from libs.save_model import *
from libs.utils import *

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

        self.dataloader = dict_DB['trainloader']
        self.s_net = dict_DB['S_Net']
        self.h_net = dict_DB['H_Net']
        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualize = dict_DB['visualize']
        self.generator = dict_DB['generator']

        self.test_process = dict_DB['test_process']

        self.eval_func = dict_DB['eval_func']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.batch_size = cfg.batch_size['img']
        self.max_iter = cfg.max_iter

        self.count = dict()
        self.count['tot'] = np.zeros(11, dtype=np.int32)
        self.count['node'] = np.zeros(11, dtype=np.int32)
        self.count['edge'] = np.zeros(11, dtype=np.int32)

        self.dataset_name = self.cfg.dataset_name.replace('_Hard', '')

    def training(self):
        loss_t = {'sum': 0, 'score': 0, 'node_loss': 0, 'edge_loss': 0}

        # train start
        self.s_net.eval()
        self.h_net.train()
        print('train start')
        logger('train start\n', self.logfile)

        for case in range(self.cfg.case_num):
            rmdir(path=self.cfg.dir['out'] + 'train/display/' + str(case))

        for i, batch in enumerate(self.dataloader):

            # load data
            img = batch['img'].cuda()
            a_idx = batch['a_idx'].cuda()
            d_idx = batch['d_idx'].cuda()
            img_name = batch['img_name'][0]

            # H_Net training data generator
            self.generator.update_batch(a_idx, d_idx,
                                        img_name=img_name, dataset_name='{}_train'.format(self.dataset_name),
                                        pickle_dir=self.cfg.dir['out'] + 'preprocess/{}_train/pickle/'.format(self.dataset_name))

            train_data1, train_data2 = self.generator.run_for_training_data(a_idx, d_idx)
            self.count['node'] += train_data1['node_count']
            self.count['edge'] += train_data2['edge_count']
            self.count['tot'] = (self.count['node'] + self.count['edge'])

            # model
            self.h_net.forward_encoder(img, train_data2, is_training=True)
            out1 = self.h_net.forward_node_score(train_data1, is_training=True)
            out2 = self.h_net.forward_edge_score(train_data2, is_training=True)
            out_f = dict(out1, **out2)

            # loss
            loss, node_loss, edge_loss = self.loss_fn(
                out=out_f,
                train_data1=train_data1,
                train_data2=train_data2)

            # optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_t['sum'] += loss.item()
            loss_t['node_loss'] += node_loss.item()
            loss_t['edge_loss'] += edge_loss.item()

            # display
            if i % self.cfg.disp_step == 0:
                print('img iter %d' % i)
                self.visualize.display_for_train_node_score(batch, out_f, train_data1, i)
                self.visualize.display_for_train_edge_score(batch, out_f, train_data2, i)
                if i % self.cfg.disp_step == 0:
                    logger("Loss : %5f, node_Loss : %5f, edge_Loss : %5f\n"
                           % (loss.item(), node_loss.item(), edge_loss.item()), self.logfile)

        # logger
        logger("Average Loss : %5f %5f %5f\n"
               % (loss_t['sum'] / i, loss_t['node_loss'] / i, loss_t['edge_loss'] / i), self.logfile)
        print("Average Loss : %5f %5f %5f\n"
               % (loss_t['sum'] / i, loss_t['node_loss'] / i, loss_t['edge_loss'] / i))

        label_list = ['tot', 'node', 'edge']
        for name in label_list:
            logger("%s label distribution\n%s\n" % (name, np.round((self.count[name] / np.sum(self.count[name])), 3) * 100), self.logfile)
            print("%s label distribution\n%s\n" % (name, np.round((self.count[name] / np.sum(self.count[name])), 3) * 100))

        # save model
        self.ckpt = {'epoch': self.epoch,
                     'model': self.h_net,
                     'optimizer': self.optimizer,
                     'val_result': self.val_result}

        save_model(checkpoint=self.ckpt,
                   param='checkpoint_final',
                   path=self.cfg.dir['weight'])

    def validation(self):
        metric = self.test_process.run(self.h_net, self.s_net, mode='val', dataset_name=self.cfg.dataset_name)

        logger("Epoch %03d => SEL HIoU %5f\n"
               % (self.ckpt['epoch'], metric['hiou']), self.logfile)
        logger("Epoch %03d => SEL EA_P %5f / EA_R %5f / EA_F %5f || EA_P_reg %5f / EA_R_reg %5f / EA_F_reg %5f \n"
               % (self.ckpt['epoch'], metric['ea_p1'], metric['ea_r1'], metric['ea_f1'], metric['ea_p2'], metric['ea_r2'], metric['ea_f2']), self.logfile)
        logger("Epoch %03d => SEL AUC_P %5f / AUC_R %5f / AUC_F %5f\n"
               % (self.ckpt['epoch'], metric['auc_p'], metric['auc_r'], metric['auc_f']), self.logfile)


        self.val_result['HIoU'] = save_model_max(self.ckpt, self.cfg.dir['weight'], self.val_result['HIoU'], metric['hiou'],
                                                  logger, self.logfile, 'HIoU', dataset_name=self.cfg.dataset_name)
        self.val_result['AUC_F'] = save_model_max(self.ckpt, self.cfg.dir['weight'], self.val_result['AUC_F'], metric['auc_f'],
                                                  logger, self.logfile, 'AUC_F', dataset_name=self.cfg.dataset_name)


    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            print('epoch %d' % epoch)
            logger("epoch %d\n" % epoch, self.logfile)

            self.training()
            if epoch > self.cfg.epoch_eval:
                self.validation()

            self.scheduler.step(self.epoch)

