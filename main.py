import os
import sys
import time
import torch
import random
import logging
import argparse
import numpy as np
from datetime import timedelta
from time import strftime, localtime
from train_eval import Instructor
from config import logger, Config, opt


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('log'):
        os.mkdir('log')
    log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log', log_file)))

    start_time = time.time()
    ins = Instructor(opt)
    ins.run()
    time_dif = get_time_dif(start_time)
    logger.info("Time usage: {}".format(time_dif))

if __name__ == '__main__':
    main()
