import pymongo
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
        print("添加一条新数据")
        return item

    def clear_files(self): 
        '''
        清空report集合所有数据
        '''
        self.coll_set.remove()

fine_grained_db = FineGrainedDb()

if __name__ == "__main__":
    item = {"_id":0, "userComment":"好吃"}
    item = {"userID":1,	"userComment":"好吃", "finegrainedLabel":[1,0,0,1], \
                        "coarsegrainedScore":[4,1,2,4], "commentScore":[1,5,2,5]}
    fine_grained_db.add_item(item)