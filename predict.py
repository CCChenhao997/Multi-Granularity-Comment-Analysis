import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import strftime, localtime
from torch.utils.data import DataLoader
from config import Config, opt
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix


class Inferer:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='./embedding/{0}_{1}_tokenizer.dat'.format(opt.model_name, opt.dataset),
                )
        embedding_matrix = build_embedding_matrix(
                vocab=self.tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='./embedding/{0}_{1}d_{2}_embedding_matrix.dat'.format(opt.model_name, str(opt.embed_dim), opt.dataset))

        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        # switch model to evaluation mode
        self.opt.predict_text = text
        predict_data = SentenceDataset(None, self.tokenizer, target_dim=self.opt.polarities_dim)
        self.test_dataloader = DataLoader(dataset=predict_data, batch_size=opt.batch_size, shuffle=False)
        self.model.eval()
        predict_tags = []
        true_tags = []
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['label'].to(self.opt.device)
                outputs = self.model(inputs)
                outputs = F.softmax(outputs, -1)
                prob, label = torch.max(outputs, -1)
                predict_prob = prob.cpu().numpy().tolist()
                predict_tags.extend(label.cpu().numpy().tolist())
                true_tags.extend(targets.cpu().numpy().tolist())

        return predict_tags, true_tags, predict_prob


# if __name__=="__main__":
infer = Inferer(opt)
# text = "第三次参加大众点评网霸王餐的活动。这家店给人整体感觉一般。首先环境只能算中等，其次霸王餐提供的菜品也不是很多，当然商家为了避免参加霸王餐吃不饱的现象，给每桌都提供了至少六份主食，我们那桌都提供了两份年糕，第一次吃火锅会在桌上有这么多的主食了。整体来说这家火锅店没有什么特别有特色的，不过每份菜品分量还是比较足的，这点要肯定！至于价格，因为没有看菜单不了解，不过我看大众有这家店的团购代金券，相当于7折，应该价位不会很高的！最后还是要感谢商家提供霸王餐，祝生意兴隆，财源广进"
# predict_tags, true_tags, predict_prob = infer.evaluate(text)
# print(predict_tags) # [0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 2, 0, 0, 0, 3, 2, 0, 0, 3, 0]
# print(true_tags)    # [0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 2, 2, 2, 2, 3, 0, 0, 0, 3, 0]
# Label Meaning = {0: 'Not mentioned', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}