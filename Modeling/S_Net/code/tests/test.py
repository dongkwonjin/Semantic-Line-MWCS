import numpy as np

from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg
        self.trainloader = dict_DB['trainloader']
        self.testloader = dict_DB['testloader']

        if 'model' in dict_DB.keys():
            self.s_net = dict_DB['S_Net']

        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.eval_line = dict_DB['eval_line']
        self.size = to_tensor(np.float32(cfg.size))
        self.visualize = dict_DB['visualize']

    def run(self, s_net, mode='val', dataset_name='SEL'):
        result = {'out': {'pos_idx': [], 'pri': [], 'mul': []},
                  'gt': {'pri': [], 'mul': []},
                  'name': []}
        datalist = []

        with torch.no_grad():
            s_net.eval()

            dataloader = self.testloader

            for i, self.batch in enumerate(dataloader):  # load batch data
                img = self.batch['img'].cuda()
                img_name = self.batch['img_name'][0]
                mul_gt = self.batch['mul_gt'][0]

                out_cls = self.forward_model.run_for_cls(img, s_net)
                self.post_process.update_data(self.batch)
                out_cls['out_pts_cls'] = self.post_process.run_for_cls(out_cls)
                out_reg = self.forward_model.run_for_reg(out_cls, s_net)
                out_reg['out_pts_reg'] = self.post_process.run_for_reg(out_reg, out_cls)
                out_f = dict(out_cls, **out_reg)

                # visualize
                if self.cfg.disp_test_result == True:
                    self.visualize.update_image(self.batch['img'][0])
                    self.visualize.update_image_name(self.batch['img_name'][0])
                    self.visualize.display_for_test(out=out_f, mul_gt=mul_gt,
                                                    idx=i, mode=mode, dataset_name=dataset_name)

                # record output data
                result['out']['pos_idx'] = (out_f['score'][0] > 0.5).nonzero()[:, 1]
                result['out']['mul'] = to_tensor(out_f['out_pts_cls'])
                result['out']['mul_reg'] = to_tensor(out_f['out_pts_reg'])
                result['gt']['mul'] = mul_gt
                result['name'] = img_name
                datalist.append(img_name)

                # save pickle
                if self.cfg.save_pickle == True:
                    dir, name = os.path.split(img_name)
                    save_pickle(dir_name=os.path.join(self.cfg.dir['out'], mode + '_' + dataset_name + '/pickle/'),
                                file_name=name[:-4],
                                data=result)

                if i % 50 == 0:
                    print('image %d ---> %s done!' % (i, img_name))

        # save pickle
        if self.cfg.save_pickle == True:
            save_pickle(dir_name=self.cfg.dir['out'] + mode + '_' + dataset_name + '/pickle/',
                        file_name='datalist',
                        data=datalist)

        # evaluation
        return self.evaluation(datalist, mode, dataset_name)

    def evaluation(self, datalist, mode, dataset_name):
        metric = dict()

        if self.cfg.eval_semantic_line == True:
            # hiou = self.eval_line.measure_HIoU(datalist, mode, dataset_name, with_reg=True)
            ea_p, ea_r, ea_f = self.eval_line.measure_EA_score(datalist, mode, dataset_name, with_reg=True)
            auc_p, auc_r, auc_f = self.eval_line.measure_AUC_PRF(datalist, mode, dataset_name, with_reg=True)

            metric['ea_p'] = ea_p
            metric['ea_r'] = ea_r
            metric['ea_f'] = ea_f

            metric['auc_p'] = auc_p
            metric['auc_r'] = auc_r

        return metric
