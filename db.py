import pymongo
import numpy as np
from collections import Counter

from config import Config, opt


class FineGrainedDb(object):

    def __init__(self):
        '''
        Args: 
        '''
        self.mongo_uri = Config.MONGO_HOST
        self.mongo_db = Config.DB_NAME
        self.mongo_port = Config.MONGO_PORT
        self.mongo_collection = Config.MONGO_COLLECTION
        self.client = pymongo.MongoClient(host=self.mongo_uri, port=self.mongo_port) # username=username, password=password)
        self.db = self.client[self.mongo_db]
        self.collection = self.db[self.mongo_collection]

    def count(self):
        '''
        查询表中记录数量
        '''
        return self.collection.find().count()
        
    def add_item(self, item):

        #exist = self.db[self.collection_name].find_one({'url': item['url']})
        self.collection.insert_one(item)
        # print("添加一条新数据")
        return item

    def clear_files(self): 
        '''
        清空该集合所有数据
        '''
        self.collection.delete_many({})

    def market_score(self):
        
        market_list = self.collection.distinct("marketName")
        market_aspect_count = {}
        market_coarse_score = {}
        market_final_score = {}
        for market in market_list:
            aspect_logit_count = {}
            corase_mean_score = {}
            
            items = []
            label_list = []
            commentScore_list = []
            coarseScore_list = []

            negative_list = []
            neutral_list = []
            positive_list = []

            items.extend(list(self.collection.find({"marketName":market})))
            for item in items:
                label_list.append(item['finegrainedLabel'])
                commentScore_list.append(item['commentScore'])
                coarseScore_list.append(item['coarsegrainedScore'])
            fine_grained_label = list(zip(*label_list))
            corase_score = list(zip(*coarseScore_list))
            # print(fine_grained_label)
            for i, score in enumerate(corase_score):
                corase_score[i] = np.mean(score) 
            for i, score in enumerate(corase_score):
                corase_mean_score[Config.aspects[i]] = score

            final_score = np.mean(commentScore_list)

            for index, aspect_label in enumerate(fine_grained_label):
                label_count = dict(Counter(aspect_label))
                label_count.setdefault(0, 0)
                label_count.setdefault(1, 0)
                label_count.setdefault(2, 0)
                label_count.setdefault(3, 0)
                label_count = sorted(label_count.items(), key = lambda x:x[0], reverse = False)
                negative_list.append(-label_count[1][1])
                neutral_list.append(label_count[2][1])
                positive_list.append(label_count[3][1])
                
                aspect_logit_count[Config.label_names[index]] = dict(label_count)
            # print(negative_list)
            negative_list.reverse()
            neutral_list.reverse()
            positive_list.reverse()

            # print(negative_list)
            market_coarse_score[market] = corase_mean_score
            market_aspect_count[market] = aspect_logit_count
            market_aspect_count[market]['negative'] = negative_list
            market_aspect_count[market]['neutral'] = neutral_list
            market_aspect_count[market]['positive'] = positive_list
            final_score = round(final_score/20)
            market_final_score[market] = final_score
        # print(market_aspect_count)
        return market_coarse_score, market_aspect_count, market_final_score

fine_grained_db = FineGrainedDb()

if __name__ == "__main__":
    # item = {"_id":0, "userComment":"好吃"}
    # item = {"userID":1,	"userComment":"好吃", "finegrainedLabel":[1,0,0,1], \
    #                     "coarsegrainedScore":[4,1,2,4], "commentScore":[1,5,2,5]}
    # fine_grained_db.add_item(item)
    fine_grained_db.market_score()
    # fine_grained_db.clear_files()