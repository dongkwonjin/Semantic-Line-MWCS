import torch
from libs.utils import *

def save_model(checkpoint, param, path):

    mkdir(path)
    torch.save({
        'epoch': checkpoint['epoch'],
        'val_result': checkpoint['val_result'],
        'model': checkpoint['model'].state_dict(),
        'optimizer': checkpoint['optimizer'].state_dict()},
        path + param)

def save_model_max(checkpoint, path, max_val, val, logger, logfile, name, dataset_name):
    if max_val < val:
        save_model(checkpoint, 'checkpoint_max_' + name + '_' + dataset_name, path)
        max_val = val
        logger("Epoch %03d => %s max %s: %5f\n" % (checkpoint['epoch'], dataset_name, name, max_val), logfile)
        print('%s == > max %s' %(dataset_name, name))

    return max_val

def save_model_max_upper(checkpoint, path, max_val, val, val2, thresd, logger, logfile, name, dataset_name):
    mkdir(path)
    if max_val < val and val2 > thresd:
        save_model(checkpoint, 'checkpoint_max_' + name + '_' + dataset_name, path)
        max_val = val
        logger("Epoch %03d => %s max %s: %5f\n" % (checkpoint['epoch'], dataset_name, name, max_val), logfile)
        print('%s == > max %s' %(dataset_name, name))

    return max_val