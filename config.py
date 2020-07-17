import sys
import logging
import torch
from models.textcnn import TextCNN

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
    }
    
    dataset_files = {
        'ai_chanllenger': {
            'train': trainset_path,
            'test': testset_path
        },
    }
    
    input_colses = {
        'textcnn': ['content'],
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