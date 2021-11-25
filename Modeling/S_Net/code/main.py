from options.config import Config
from tests.test import *
from trains.train import *
from libs.prepare import *


def main_eval(cfg, dict_DB):

    # test option
    test_process = Test_Process(cfg, dict_DB)
    datalist = load_pickle(cfg.dir['out'] + 'test_' + cfg.dataset_name + '/pickle/datalist')
    test_process.evaluation(datalist, mode='test', dataset_name=cfg.dataset_name)

def main_test(cfg, dict_DB):

    # test option
    test_process = Test_Process(cfg, dict_DB)
    test_process.run(dict_DB['S_Net'], mode='test', dataset_name=cfg.dataset_name)

def main_train(cfg, dict_DB):

    # train option
    dict_DB['test_process'] = Test_Process(cfg, dict_DB)
    train_process = Train_Process(cfg, dict_DB)
    train_process.run()

def main():

    # Config
    cfg = Config()

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)
    dict_DB = prepare_model(cfg, dict_DB)
    dict_DB = prepare_evaluation(cfg, dict_DB)
    dict_DB = prepare_post_processing(cfg, dict_DB)
    dict_DB = prepare_training(cfg, dict_DB)
    # dict_DB = prepare_generator(cfg, dict_DB)


    if 'test' in cfg.run_mode:
        main_test(cfg, dict_DB)
    if 'train' in cfg.run_mode:
        main_train(cfg, dict_DB)
    if 'eval' in cfg.run_mode:
        main_eval(cfg, dict_DB)

if __name__ == '__main__':
    main()
