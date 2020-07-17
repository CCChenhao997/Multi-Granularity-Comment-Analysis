import os
import torch
import torch.nn as nn
import argparse
import pandas as pd
from time import strftime, localtime
from torch.utils.data import DataLoader
from models.textcnn import TextCNN
from config import Config
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix


class Inferer:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt
        self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
                max_length=opt.max_length, 
                data_file='{0}_tokenizer.dat'.format(opt.dataset),
                opt=opt)
        embedding_matrix = build_embedding_matrix(
                vocab=self.tokenizer.vocab, 
                embed_dim=opt.embed_dim, 
                data_file='{0}d_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))

        self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        # switch model to evaluation mode
        self.opt.predict_text = text
        predict_data = SentenceDataset(None, self.tokenizer, target_dim=self.opt.polarities_dim, opt=self.opt)
        self.test_dataloader = DataLoader(dataset=predict_data, batch_size=opt.batch_size, shuffle=False)
        self.model.eval()
        predict_tags = []
        true_tags = []
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['label'].to(self.opt.device)
                outputs = self.model(inputs)
                predict_tags.extend(torch.argmax(outputs, -1).cpu().numpy().tolist())
                true_tags.extend(targets.cpu().numpy().tolist())

        return predict_tags, true_tags


# if __name__=="__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='textcnn', type=str, help=', '.join(Config.model_classes.keys()))
parser.add_argument('--dataset', default='ai_chanllenger', type=str, help=', '.join(Config.dataset_files.keys()))
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
parser.add_argument('--filter_sizes', default=(2, 3, 4), type=tuple)
parser.add_argument('--num_filters', default=256, type=int)
parser.add_argument('--seed', default=1000, type=int)
parser.add_argument('--predict_text', default=None, type=str)
opt = parser.parse_args()
    
opt.model_class = Config.model_classes[opt.model_name]
opt.dataset_file = Config.dataset_files[opt.dataset]
opt.inputs_cols = Config.input_colses[opt.model_name]
opt.state_dict_path = Config.model_state_dict_paths[opt.model_name]
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)

infer = Inferer(opt)
# text = "第三次参加大众点评网霸王餐的活动。这家店给人整体感觉一般。首先环境只能算中等，其次霸王餐提供的菜品也不是很多，当然商家为了避免参加霸王餐吃不饱的现象，给每桌都提供了至少六份主食，我们那桌都提供了两份年糕，第一次吃火锅会在桌上有这么多的主食了。整体来说这家火锅店没有什么特别有特色的，不过每份菜品分量还是比较足的，这点要肯定！至于价格，因为没有看菜单不了解，不过我看大众有这家店的团购代金券，相当于7折，应该价位不会很高的！最后还是要感谢商家提供霸王餐，祝生意兴隆，财源广进"
# predict_tags, true_tags = infer.evaluate(text)
# print(predict_tags)
# print(true_tags)
# Label Meaning = {0: 'Not mentioned', 1: 'Negative', 2: 'Neutral', 3: 'Positive'}