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
from models.textcnn import TextCNN
from train_eval import Instructor
from config import logger, Config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='textcnn', type=str, help=', '.join(Config.model_classes.keys()))
    parser.add_argument('--dataset', default='ai_chanllenger', type=str, help=', '.join(Config.dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(Config.optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(Config.initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)    # 1e-3
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)    # 1e-5
    parser.add_argument('--num_epoch', default=5, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--polarities_dim', default=4, type=int)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--max_length', default=500, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--repeats', default=1, type=int)
    parser.add_argument('--filter_sizes', default=(2, 3, 4), type=tuple)
    parser.add_argument('--num_filters', default=256, type=int)
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--predict_text', default=None, type=str)
    opt = parser.parse_args()
    	
    opt.model_class = Config.model_classes[opt.model_name]
    opt.dataset_file = Config.dataset_files[opt.dataset]
    opt.inputs_cols = Config.input_colses[opt.model_name]
    opt.initializer = Config.initializers[opt.initializer]
    opt.optimizer = Config.optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
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
