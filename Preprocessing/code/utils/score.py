import torch

def overlap_score(ref_mask, tar_mask):

    '''
    :param ref_img: HxW
    :return:
    '''

    score = ref_mask * tar_mask
    score = torch.sum(score, dim=(1, 2)) / torch.sum(ref_mask, dim=(0, 1))

    return score



# def overlap_score(ref_img, line_mask):
#
#     '''
#     :param ref_img: HxW
#     :return:
#     '''
#
#     score = ref_img * line_mask
#     score = torch.sum(score, dim=(1, 2)) / torch.sum(line_mask, dim=(1, 2))
#
#     return score
