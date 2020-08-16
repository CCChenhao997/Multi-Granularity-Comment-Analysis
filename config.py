import sys
import logging
import torch
from models.textcnn import TextCNN
from models.gcae import GCAE

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Config:
    
    label_meaning = {0: 'Not mentioned', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}

    label_trans = lambda x: Config.label_meaning[x]

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
    }