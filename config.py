import os
import sys
import logging
import torch
import argparse
from models.textcnn import TextCNN
from models.gcae import GCAE

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Config:
    
    label_meaning = {0: 'Not mentioned', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

    label_trans = lambda x: Config.label_meaning[x]

    stopwordsList = ['\n', ' ', '【', '】', '（', '）', '#']
    stopwords ='|'.join(stopwordsList)

    embedding_path = './data/word_embedding/sgns.sogou.char'
    trainset_path = './data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv'
    testset_path = './data/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv'

    label_names = ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find',
                'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
                'price_level', 'price_cost_effective', 'price_discount', 'environment_decoration', 'environment_noise',
                'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look',
                'dish_recommendation','others_overall_experience', 'others_willing_to_consume_again']
    
    label_chinese_name = ['交通是否便利', '距离商圈远近', '是否容易寻找',
                          '排队等候时间', '服务人员态度', '是否容易停车', '点菜上菜速度',
                          '价格水平', '性价比', '折扣力度',
                          '装修情况', '嘈杂情况', '就餐空间', '卫生情况',
                          '菜品分量', '菜品口感', '菜品外观', '菜品推荐程度',
                          '本次消费感受', '再次消费意愿']

    aspect_names = {
                    'location': ['location_traffic_convenience', 'location_distance_from_business_district', 'location_easy_to_find'], 
                    'service': ['service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed'], 
                    'price': ['price_level', 'price_cost_effective', 'price_discount'], 
                    'environment': ['environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness'], 
                    'dish': ['dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation'], 
                    'others': ['others_overall_experience', 'others_willing_to_consume_again']
                    }

    
    model_classes = {
        'textcnn': TextCNN,
        'gcae': GCAE,
    }
    
    dataset_files = {
        'ai_chanllenger': {
            'train': trainset_path,
            'test': testset_path
        },
    }
    
    input_colses = {
        'textcnn': ['content'],
        'gcae': ['content', 'aspect']
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,    # default lr=0.01
        'adam': torch.optim.Adam,          # default lr=0.001
        'adamax': torch.optim.Adamax,      # default lr=0.002
        'asgd': torch.optim.ASGD,          # default lr=0.01
        'rmsprop': torch.optim.RMSprop,    # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    model_state_dict_paths = {
        'textcnn': './state_dict/textcnn_ai_chanllenger_4class_acc0.8298',
        'gcae': './state_dict/gcae_ai_chanllenger_4class_acc0.8101',
    }


# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='gcae', type=str, help=', '.join(Config.model_classes.keys()))
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
parser.add_argument('--aspect_maxlen', default=6, type=int)
parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
parser.add_argument('--repeats', default=1, type=int)
parser.add_argument('--filter_sizes', default=(2, 3, 4), type=tuple)
parser.add_argument('--num_filters', default=100, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--cuda', default='0', type=str)
parser.add_argument('--predict_text', default=None, type=str)
parser.add_argument('--sample', default=False, action='store_true')
opt = parser.parse_args()
    
opt.model_class = Config.model_classes[opt.model_name]
opt.dataset_file = Config.dataset_files[opt.dataset]
opt.inputs_cols = Config.input_colses[opt.model_name]
opt.initializer = Config.initializers[opt.initializer]
opt.optimizer = Config.optimizers[opt.optimizer]
opt.state_dict_path = Config.model_state_dict_paths[opt.model_name]
os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)