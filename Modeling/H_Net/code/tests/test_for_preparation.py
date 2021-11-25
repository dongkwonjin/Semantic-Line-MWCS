import numpy as np

from libs.utils import *

class Test_Process_for_preparation(object):
    def __init__(self, cfg, dict_DB):

        self.cfg = cfg

        self.trainloader = dict_DB['trainloader']
        self.testloader = dict_DB['testloader']

        self.forward_model = dict_DB['forward_model']
        self.post_process = dict_DB['post_process']
        self.size = to_tensor(np.float32(cfg.size))
        self.visualize = dict_DB['visualize']

        self.candidates = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates')

    def run(self, s_net, mode='val', dataset_name='SEL', max_iter=None):
        print('preprocess start: generating training data for H-Net')
        logger('train start\n', self.logfile)

        result = {'out': {'pri': [], 'mul': []},
                  'gt': {'pri': [], 'mul': []},
                  'name': []}
        datalist = []

        with torch.no_grad():
            s_net.eval()
            for i, self.batch in enumerate(self.trainloader):  # load batch data

                img = self.batch['img'].cuda()
                img_name = self.batch['img_name'][0]
                mul_gt = self.batch['mul_gt'][0]

                out, _ = self.forward_model.generate_node(img, s_net, max_iter)
                out['cls_mul_pts'] = self.candidates[to_np(out['center_idx'])]
                # visualize
                if self.cfg.disp_test_result == True:
                    self.visualize.update_image(self.batch['img'][0])
                    self.visualize.update_image_name(self.batch['img_name'][0])
                    self.visualize.display_for_test(out=out, mul_gt=mul_gt, idx=i,
                                                    mode=mode, dataset_name=dataset_name)

                # record output data
                result['out']['cls_idx'] = (out['score'][0, 0] > 0.5).nonzero()[:, 0]
                result['out']['center_idx'] = out['center_idx']
                result['out']['cluster_idx'] = out['cluster_idx']
                result['out']['mul'] = to_tensor(out['cls_mul_pts'])
                result['gt']['mul'] = mul_gt
                result['name'] = img_name
                datalist.append(img_name)

                # save pickle
                if self.cfg.save_pickle == True:
                    dir, name = os.path.split(img_name)
                    save_pickle(dir_name='{}preprocess/{}/pickle/'.format(self.cfg.dir['out'], dataset_name), file_name=name[:-4], data=result)
                if i % 50 == 0:
                    print('image %d ---> %s done!' % (i, img_name))

        # save pickle
        if self.cfg.save_pickle == True:
            save_pickle(dir_name='{}preprocess/{}/pickle/'.format(self.cfg.dir['out'], dataset_name), file_name='datalist', data=datalist)

