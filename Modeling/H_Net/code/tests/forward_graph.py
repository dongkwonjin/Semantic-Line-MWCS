from libs.utils import *
from libs.module_for_region_pooling import *

class Forward_Model(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.sep_region = Separated_Region(cfg)

    def generate_node(self, img, model, max_iter):
        out_temp = {}
        out_temp['center_idx'] = torch.LongTensor([]).cuda()
        out_temp['cluster_idx'] = torch.LongTensor([]).cuda()

        model.initialize()
        out = model.forward_for_cls(img, is_training=False)
        score = out['score'].clone()

        for i in range(max_iter):

            # Clustering
            score_temp = score * model.visit_mask
            cluster_mask, cluster, max_score = model.selection_and_removal(score_temp)

            center_idx = cluster['center_idx']
            cluster_idx = cluster['cluster_idx']

            if self.cfg.constrain_max_score == True:
                if i >= 2 and max_score < 0.5:
                    break

            if i == 0:
                out_temp['primary_idx'] = center_idx.clone()

            out_temp['center_idx'] = torch.cat((out_temp['center_idx'], center_idx), dim=0)
            out_temp['cluster_idx'] = torch.cat((out_temp['cluster_idx'], cluster_idx), dim=0)
            model.center_mask[:, :, center_idx] = 0
            model.visit_mask[:, :, cluster_idx] = 0


        node_data = dict()
        node_idx_set = out_temp['center_idx'].unsqueeze(1)
        node_data['node_idx_set'] = node_idx_set
        node_data['line_data'] = self.sep_region.run(node_idx_set[:, 0])

        out_temp['score'] = score

        return out_temp, node_data

    def generate_edge(self, out):

        node = out['center_idx']

        num = node.shape[0]
        idx_set1 = []
        idx_set2 = []
        for i in range(num):
            for j in range(i + 1, num):
            # for j in range(i, num):
                idx_set1.append(int(node[i]))
                idx_set2.append(int(node[j]))
        idx_set1 = to_tensor(np.array(idx_set1)).unsqueeze(1)
        idx_set2 = to_tensor(np.array(idx_set2)).unsqueeze(1)

        edge_data = dict()
        edge_idx_set = torch.cat((idx_set1, idx_set2), dim=1)
        edge_data['num'] = (num) * (num - 1) // 2
        edge_data['line_data1'] = self.sep_region.run(edge_idx_set[:, 0])
        edge_data['line_data2'] = self.sep_region.run(edge_idx_set[:, 1])
        edge_data['edge_idx_set'] = edge_idx_set


        return edge_data

