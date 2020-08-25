# import os
# import re
import json
import numpy as np
from loguru import logger
from predict import infer
from config import Config
from db import fine_grained_db
from flask import Flask, request, jsonify, Response

app = Flask(__name__)

@app.route("/sa", methods=['POST'])
def sa_predict():
    try:
        if request.method == 'POST':
            input_data = request.get_data()
            logger.info("接受到一个SA post请求, IP:{}.".format(request.remote_addr))
            if isinstance(input_data, bytes):
                input_data = str(input_data, encoding='utf-8')
            logger.info('input_data:{}'.format(input_data))
            input_json = json.loads(input_data)
            predict_tags, _, predict_prob = infer.evaluate(input_json['text'])
            # predict_tags = predict_tags[0]

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

            radarScore = [i*100 for i in predict_tags]
            predict_tags_trans = [Config.label_trans(i) for i in predict_tags]
            aspect_layer2_dict = dict(zip(Config.label_names, predict_tags_trans))
            
            data = {'Aspect_first_layer': aspect_layer1_dict, 'Aspect_second_layer': aspect_layer2_dict, \
                    'scoreList': scoreList, 'radarScore': radarScore}
            print(data)
            # return jsonify({'predict_tags': aspect_label})
            if Config.mongo_save:
                num = fine_grained_db.count()
                comment_score = np.mean(scoreList)
                item = {"_id":num+1, "userID":num+1, "userName":input_json['customerName'], "userGender":input_json['customerGender'],  \
                        "userComment":input_json['text'], "marketName":input_json['resName'], \
                        "finegrainedLabel":predict_tags, "coarsegrainedScore":scoreList, "commentScore":comment_score}
                fine_grained_db.add_item(item)
                
            return Response(json.dumps(data, sort_keys=False, indent=4, ensure_ascii=False), mimetype='application/json')
        else:
            logger.info("客户端提交了一个get请求，应该为post请求")
            return "YOU NEED USE POST ", 500

    except Exception as e:
        logger.debug("{}:{}".format(type(e), e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7777)
