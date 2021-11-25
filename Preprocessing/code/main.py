import torch.nn.parallel
import torch.optim

from options.config import Config
from libs.prepare import *
from libs.utils import *
from libs.line_generator import *


def run_line_generator(cfg, dict_DB):

    line_generator = generate_line(cfg, dict_DB)
    line_generator.run()

def main():

    # option
    cfg = Config()

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    torch.backends.cudnn.deterministic = True

    # prepare
    dict_DB = dict()
    dict_DB = prepare_visualization(cfg, dict_DB)
    dict_DB = prepare_dataloader(cfg, dict_DB)

    # run
    run_line_generator(cfg, dict_DB)

if __name__ == '__main__':
    main()
