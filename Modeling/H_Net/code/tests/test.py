import numpy as np

from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg

        self.trainloader = dict_DB['trainloader']
        self.testloader = dict_DB['testloader']

        if 'model' in dict_DB.keys():
            self.model = dict_DB['H_Net']

        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.eval_line = dict_DB['eval_line']
        self.visualize = dict_DB['visualize']

    def run(self, h_net, s_net, mode='val', dataset_name='SEL'):
        result = {'out': {'pri': [], 'mul': []},
                  'gt': {'pri': [], 'mul': []},
                  'name': []}
        datalist = []

        with torch.no_grad():
            s_net.eval()
            h_net.eval()

            for i, self.batch in enumerate(self.testloader):  # load batch data

                img = self.batch['img'].cuda()
                img_name = self.batch['img_name'][0]
                mul_gt = self.batch['mul_gt'][0]

                out_s, node_data = self.forward_model.generate_node(img, s_net, self.cfg.max_iter)
                edge_data = self.forward_model.generate_edge(out_s)

                h_net.forward_encoder(img, edge_data, is_training=False)
                out_h1 = h_net.forward_node_score(node_data, is_training=False)
                out_h2 = h_net.forward_edge_score(edge_data, is_training=False)

                out = dict(out_s, **out_h1, **out_h2)
                self.post_process.update(self.batch, out, mode, dataset_name, i)
                out_cls = self.post_process.run()
                out_reg = s_net.forward_for_reg(idx=out_cls['cls_idx'])
                out_reg = self.post_process.run_for_reg(out_reg, out_cls)
                out_f = dict(out, **out_cls, **out_reg)

                # visualize
                if self.cfg.disp_test_result == True:
                    self.visualize.update_image(self.batch['img'][0])
                    self.visualize.update_image_name(self.batch['img_name'][0])
                    self.visualize.display_for_test(out=out_f, mul_gt=mul_gt, idx=i,
                                                    mode=mode, dataset_name=dataset_name)

                # record output data
                result['out']['pri_cls'] = out_f['cls_pri_pts']
                result['out']['mul_cls'] = out_f['cls_mul_pts']
                result['out']['pri_reg'] = to_tensor(out_f['reg_pri_pts'])
                result['out']['mul_reg'] = to_tensor(out_f['reg_mul_pts'])
                result['gt']['mul'] = mul_gt
                result['name'] = img_name
                datalist.append(img_name)

                # save pickle
                if self.cfg.save_pickle == True:
                    dir, name = os.path.split(img_name)
                    save_pickle(dir_name='{}{}_{}/pickle/'.format(self.cfg.dir['out'], mode, dataset_name), file_name=name[:-4], data=result)
                if i % 50 == 0:
                    print('image %d ---> %s done!' % (i, img_name))

        # save pickle
        if self.cfg.save_pickle == True:
            save_pickle(dir_name='{}{}_{}/pickle/'.format(self.cfg.dir['out'], mode, dataset_name),
                        file_name='datalist',
                        data=datalist)

        # evaluation
        return self.evaluation(datalist, mode, dataset_name)

    def evaluation(self, datalist, mode, dataset_name):
        metric = dict()

        if self.cfg.eval_semantic_line == True:
            hiou = self.eval_line.measure_HIoU(datalist, mode, dataset_name, with_reg=True)
            ea_p1, ea_r1, ea_f1 = self.eval_line.measure_EA_score(datalist, mode, dataset_name, with_reg=False)
            ea_p2, ea_r2, ea_f2 = self.eval_line.measure_EA_score(datalist, mode, dataset_name, with_reg=True)
            auc_p, auc_r, auc_f = self.eval_line.measure_AUC_PRF(datalist, mode, dataset_name, with_reg=True)

            metric['hiou'] = hiou

            metric['ea_p1'] = ea_p1
            metric['ea_r1'] = ea_r1
            metric['ea_f1'] = ea_f1

            metric['ea_p2'] = ea_p2
            metric['ea_r2'] = ea_r2
            metric['ea_f2'] = ea_f2

            metric['auc_p'] = auc_p
            metric['auc_r'] = auc_r
            metric['auc_f'] = auc_f

        return metric
