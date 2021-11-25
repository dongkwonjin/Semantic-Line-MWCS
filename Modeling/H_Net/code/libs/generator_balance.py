import random
import math

# utils
from libs.utils import *
from libs.modules import *

from libs.module_for_region_pooling import *

class Generate_Training_Data(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.height = cfg.height
        self.width = cfg.width
        self.max_dist = self.cfg.max_dist
        self.center = np.array(self.cfg.center_pt)

        self.candidates = to_tensor(load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'candidates'))
        hough_space = load_pickle(self.cfg.dir['preprocess'][self.cfg.dataset_name] + 'hough_space')
        self.hough_space = hough_space
        self.hough_h, self.hough_w = self.hough_space['idx'].shape
        self.hough_space_idx_all = to_tensor(self.hough_space['idx'])
        self.hough_space_idx = to_tensor(self.hough_space['idx'][self.hough_h // 3:self.hough_h // 3 * 2])
        self.angle_list = to_tensor(np.float32(self.hough_space['angle_list'])) / 90
        self.dist_list = to_tensor(np.float32(self.hough_space['dist_list'])) / self.max_dist

        self.cand_idx = torch.arange(0, torch.max(self.hough_space_idx)+1)
        self.cand_num = self.cand_idx.shape[0]
        self.cluster_num = self.cfg.cluster_num

        self.sigma_a = 1.5
        self.sigma_d = 1.5

        self.batch_size_cls = 30

        self.sep_region = Separated_Region(cfg)

    def run_for_training_data(self, a_idx, d_idx):
        node_idx_set = torch.LongTensor([]).cuda()
        edge_idx_set = torch.LongTensor([]).cuda()
        node_score = torch.FloatTensor([]).cuda()
        node_case = torch.LongTensor([]).cuda()
        edge_score = torch.FloatTensor([]).cuda()
        edge_case = torch.LongTensor([]).cuda()
        node_count = np.zeros(11, dtype=np.int32)
        edge_count = np.zeros(11, dtype=np.int32)

        # gt idx & neg idx
        gt_num = a_idx.shape[1]

        # pos idx from detector
        check_pos = torch.zeros(self.cand_num, dtype=torch.long).cuda()
        check_pos[self.detector_pos_idx] = 1

        for i in range(self.batch_size_cls):

            for cls in range(2):
                flag = -1
                if cls == 1:  # different pos line
                    case = random.randint(0, 2)

                    if case != 2 and gt_num > 1:  # different pos line
                        rand_idx_list = torch.randperm(gt_num)[:2]
                        gt_idx1 = self.gt_grid_idx['1'][rand_idx_list[0]]
                        gt_idx2 = self.gt_grid_idx['1'][rand_idx_list[1]]
                        hough_idx1 = (self.hough_space_idx == gt_idx1).nonzero()
                        hough_idx2 = (self.hough_space_idx == gt_idx2).nonzero()
                        dist = torch.norm((hough_idx1 - hough_idx2).type(torch.float))

                        if dist <= 3:
                            grid_size = '1'
                        elif case == 0:
                            grid_size = '5'
                        elif case == 1:
                            grid_size = '3'

                        pos_idx_list1 = self.gt_grid_idx[grid_size][rand_idx_list[0]]
                        pos_score_list1 = self.gt_grid_score[grid_size][rand_idx_list[0]]
                        rand_idx = torch.randperm(pos_idx_list1.shape[0])
                        pos_idx1 = pos_idx_list1[rand_idx[0]].view(1, 1)
                        score1 = pos_score_list1[rand_idx[0]]

                        pos_idx_list2 = self.gt_grid_idx[grid_size][rand_idx_list[1]]
                        pos_score_list2 = self.gt_grid_score[grid_size][rand_idx_list[1]]
                        rand_idx = torch.randperm(pos_idx_list2.shape[0])
                        pos_idx2 = pos_idx_list2[rand_idx[0]].view(1, 1)
                        score2 = pos_score_list2[rand_idx[0]]

                        hough_idx1 = (self.hough_space_idx == pos_idx1).nonzero()
                        hough_idx2 = (self.hough_space_idx == pos_idx2).nonzero()
                        dist = torch.norm((hough_idx1 - hough_idx2).type(torch.float))

                        if dist <= 3:
                            continue

                        else:
                            edge_idx = torch.cat((pos_idx1, pos_idx2), dim=1)
                            score = (score1 + score2) / 2
                        flag = 'edge'

                    elif case == 2 or gt_num >= 1:
                        rand_idx_list = torch.randperm(gt_num)[:1]
                        pos_idx_list = self.gt_grid_idx['3'][rand_idx_list[0]]
                        pos_score_list = self.gt_grid_score['3'][rand_idx_list[0]]
                        rand_idx = torch.randperm(pos_idx_list.shape[0])
                        node_idx = pos_idx_list[rand_idx[0]].view(1, 1)
                        score = pos_score_list[rand_idx[0]]

                        flag = 'node'

                else:
                    case = random.randint(3, 6)
                    randomness = random.randint(0, 1)
                    # randomness = 1
                    if self.dataset_name == 'test_SEL':
                        randomness = 1

                    if randomness == 0:
                        check = torch.ones(self.cand_num, dtype=torch.long).cuda()
                    else:
                        check = check_pos.clone()

                    if case == 3:  # two different lines

                        excluded_pos_idx = torch.randperm(gt_num)
                        for j in range(gt_num):
                            ep_idx = excluded_pos_idx[j]
                            check[self.gt_grid_idx['5'][ep_idx]] = 2

                        neg_idx_list = self.cand_idx[check == 1].cuda()
                        rand_idx_list = torch.randperm(neg_idx_list.shape[0])
                        edge_idx = neg_idx_list[rand_idx_list[:2]].view(1, -1)

                        flag = 'edge'

                    elif case == 4:  # two different lines with pos

                        rand_idx_list = torch.randperm(gt_num)
                        pos_idx = rand_idx_list[:1]

                        check[self.gt_grid_idx['5'][pos_idx]] = 2

                        excluded_pos_idx = rand_idx_list[1:]
                        for j in range(gt_num - 1):
                            ep_idx = excluded_pos_idx[j]
                            check[self.gt_grid_idx['5'][ep_idx]] = 3

                        pos_idx_list = (check == 2).nonzero()
                        rand_idx_list = torch.randperm(pos_idx_list.shape[0])
                        neg_idx1 = pos_idx_list[rand_idx_list[:1]].view(1, -1)

                        neg_idx_list = (check == 1).nonzero()
                        rand_idx_list = torch.randperm(neg_idx_list.shape[0])
                        neg_idx2 = neg_idx_list[rand_idx_list[:1]].view(1, -1)

                        edge_idx = torch.cat((neg_idx1, neg_idx2), dim=1)

                        flag = 'edge'

                    elif case == 5:  # overlapping pos pair

                        rand_idx_list = torch.randperm(gt_num)
                        pos_idx = rand_idx_list[:1]

                        check[self.gt_grid_idx['7'][pos_idx]] = 2

                        excluded_pos_idx = rand_idx_list[1:]
                        for j in range(gt_num - 1):
                            ep_idx = excluded_pos_idx[j]
                            check[self.gt_grid_idx['3'][ep_idx]] = 3

                        pos_idx_list = (check == 2).nonzero()
                        rand_idx_list = torch.randperm(pos_idx_list.shape[0])

                        edge_idx = pos_idx_list[rand_idx_list[:2]].view(1, -1)

                        flag = 'edge'

                    elif case == 6:
                        for j in range(gt_num):
                            check[self.gt_grid_idx['7'][j]] = 2

                        neg_idx_list = self.cand_idx[check == 1].cuda()
                        rand_idx_list = torch.randperm(neg_idx_list.shape[0])
                        node_idx = neg_idx_list[rand_idx_list[:1]].view(1, -1)

                        flag = 'node'

                    score = 0

                if flag == 'node':
                    if node_idx.shape[1] == 0:
                        continue

                    node_idx_set = torch.cat((node_idx_set, node_idx), dim=0)
                    node_score = torch.cat((node_score, torch.FloatTensor([score]).cuda()), dim=0)
                    node_case = torch.cat((node_case, torch.LongTensor([case]).cuda()), dim=0)
                    node_count[int(np.round(float(score), 1) * 10)] += 1
                elif flag == 'edge':
                    if edge_idx.shape[1] < 2:
                        continue

                    rand_flip = torch.randperm(2)
                    edge_idx_set = torch.cat((edge_idx_set, edge_idx[:, rand_flip]), dim=0)
                    edge_score = torch.cat((edge_score, torch.FloatTensor([score]).cuda()), dim=0)
                    edge_case = torch.cat((edge_case, torch.LongTensor([case]).cuda()), dim=0)
                    edge_count[int(np.round(float(score), 1) * 10)] += 1

        # shuffle
        num = edge_case.shape[0]
        rand_idx_list = torch.randperm(num)
        edge_idx_set = edge_idx_set[rand_idx_list]
        edge_score = edge_score[rand_idx_list]
        edge_case = edge_case[rand_idx_list]

        num = node_case.shape[0]
        rand_idx_list = torch.randperm(num)
        node_idx_set = node_idx_set[rand_idx_list]
        node_score = node_score[rand_idx_list]
        node_case = node_case[rand_idx_list]


        node_data = dict()
        node_data['node_idx_set'] = node_idx_set
        node_data['line_data'] = self.sep_region.run(node_idx_set[:, 0])
        node_data['gt_node_score'] = node_score.view(-1, 1)
        node_data['node_case'] = node_case
        node_data['node_count'] = node_count


        edge_data = dict()
        edge_data['edge_idx_set'] = edge_idx_set
        edge_data['line_data1'] = self.sep_region.run(edge_idx_set[:, 0])
        edge_data['line_data2'] = self.sep_region.run(edge_idx_set[:, 1])
        edge_data['gt_edge_score'] = edge_score.view(-1, 1)
        edge_data['edge_case'] = edge_case
        edge_data['edge_count'] = edge_count

        return node_data, edge_data

    def generate_grid(self, a_idx, d_idx):
        gt_num = a_idx.shape[1]

        self.gt_grid_idx = {'1': [], '3': [], '5': [], '7': [], '7_d': []}
        self.gt_grid_score = {'1': [], '3': [], '5': [], '7': [], '7_d': []}
        for i in range(gt_num):
            region_idx, weight = self.adjacent_grid_idx(a_idx[0, i], d_idx[0, i], 0, 0)
            self.gt_grid_idx['1'].append(region_idx)
            self.gt_grid_score['1'].append(weight)
            region_idx, weight = self.adjacent_grid_idx(a_idx[0, i], d_idx[0, i], 1, 1)
            self.gt_grid_idx['3'].append(region_idx)
            self.gt_grid_score['3'].append(weight)
            region_idx, weight = self.adjacent_grid_idx(a_idx[0, i], d_idx[0, i], 2, 2)
            self.gt_grid_idx['5'].append(region_idx)
            self.gt_grid_score['5'].append(weight)
            region_idx, weight = self.adjacent_grid_idx(a_idx[0, i], d_idx[0, i], 3, 3)
            self.gt_grid_idx['7'].append(region_idx)
            self.gt_grid_score['7'].append(weight)


    def adjacent_grid_idx(self, a_idx, d_idx, da, dd):
        h, w = self.hough_space['idx'].shape
        x1 = np.maximum(to_np(a_idx) - da, 0)
        x2 = np.minimum(to_np(a_idx) + da + 1, h)
        y1 = np.maximum(to_np(d_idx) - dd, 0)
        y2 = np.minimum(to_np(d_idx) + dd + 1, w)

        dx = x2 - x1
        dy = y2 - y1
        X, Y = torch.meshgrid(torch.arange(x1, x2).cuda(), torch.arange(y1, y2).cuda())
        X = a_idx - X.view(dx, dy, 1)
        Y = d_idx - Y.view(dx, dy, 1)
        diff_idx = torch.cat((X, Y), dim=2).view(-1, 2)
        weight = torch.exp(-1 * (torch.pow(diff_idx, 2)[:, 0] / (2 * self.sigma_a) + torch.pow(diff_idx, 2)[:, 1] / (2 * self.sigma_d)))
        region_idx = to_tensor(self.hough_space['idx'][x1:x2, y1:y2]).type(torch.long).reshape(-1)

        return region_idx, weight

    def update_batch(self, a_idx, d_idx, img_name, dataset_name, pickle_dir, img=None, visualize=None):
        self.img = img
        self.img_name = img_name
        self.visualize = visualize
        self.dataset_name = dataset_name

        pickle_path = pickle_dir + img_name[:-4]
        data = load_pickle(pickle_path)
        self.detector_pos_idx = data['out']['cluster_idx']
        self.generate_grid(a_idx, d_idx)

