import re
import pandas as pd 
import numpy as np
from tqdm import tqdm


from config import Config, opt
from db import fine_grained_db
from predict import infer

def parse_data(data_path):
    df = pd.read_csv(data_path, encoding='utf-8')
    data = []
    for index, line in tqdm(df.iterrows()):
        content = line[1].strip().strip('"')
        content = re.sub(Config.stopwords, '', content)
        if len(content) > opt.max_length:
            continue
        data.append(content)
    return data
    # print(len(data)) # * 91641条数据

def add_data(data_path):
    data = parse_data(data_path)
    market_name = ["南京大排档", "多伦多海鲜自助", "探鱼","新石器烤肉","海底捞" ]
    for index, input_data in tqdm(enumerate(data)):
        predict_tags, _, predict_prob = infer.evaluate(input_data)

        base = 0
        aspect_layer1_dict = {}
        scoreList = []
        for aspect in Config.aspect_names:
            # tags = []
            score = 80
            positive_score = (100 - score) / len(Config.aspect_names[aspect])
            neutral_score = positive_score / 2
            negative_score = -(100 - score) / len(Config.aspect_names[aspect])
            for i in range(base, len(Config.aspect_names[aspect]) + base):

                if predict_tags[i] == 3:
                    score += positive_score * predict_prob[i]
                elif predict_tags[i] == 2:
                    score += neutral_score * predict_prob[i]
                elif predict_tags[i] == 1:
                    score += negative_score * predict_prob[i]
                else:
                    pass

                # tags.append(predict_tags[i])
            score = round(score, 2)
            aspect_layer1_dict.update({aspect: score})
            base += len(Config.aspect_names[aspect])
            scoreList.append(score)
            # score_list.insert(0, score)

        # radarScore = [i*100 for i in predict_tags]
        # predict_tags_trans = [Config.label_trans(i) for i in predict_tags]
        # aspect_layer2_dict = dict(zip(Config.label_names, predict_tags_trans))
        
        # data = {'Aspect_first_layer': aspect_layer1_dict, 'Aspect_second_layer': aspect_layer2_dict, \
        #         'scoreList': scoreList, 'radarScore': radarScore}
        
        num = fine_grained_db.count()
        comment_score = np.mean(scoreList)
        if index <= len(data)/5:
            name = market_name[0]
        elif index > len(data)/5 and index <= 2/5 * len(data):
            name = market_name[1]
        elif index > 2 * len(data)/5 and index <= 3/5 * len(data):
            name = market_name[2]
        elif index > 3 * len(data)/5 and index <= 4/5 * len(data):
            name = market_name[3]
        else:
            name = market_name[4]
        item = {"_id":num+1, "userID":num+1, "userName":"RAO", "userGender":"1",  \
                "userComment":input_data, "marketName":name, \
                "finegrainedLabel":predict_tags, "coarsegrainedScore":scoreList, "commentScore":comment_score}
        fine_grained_db.add_item(item)



if __name__ == "__main__":
    add_data('data/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')
