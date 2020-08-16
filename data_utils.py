import os
import re
import pickle
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from config import Config


# Meaning	    Positive	Neutral	    Negative	Not mentioned
# Old labels    1	        0	        -1	        -2
# New labels    3           2           1           0
def map_sentimental_type(value):
    return value + 2

def parse_data_for_textcnn(data_path, predict_text=None):
    all_data = []
    if predict_text is not None:
        content = predict_text.strip()
        label = []
        for idx, name in enumerate(Config.label_names):
            label.append(0)
        data = {'content': content, 'label': label}
        all_data.append(data)

    else:
        df = pd.read_csv(data_path, encoding='utf-8')
        for index, line in tqdm(df.iterrows()):
            content = line[1].strip()
            label = []
            for idx, name in enumerate(Config.label_names):
                polarity = map_sentimental_type(int(line[idx+2]))
                label.append(polarity)

            data = {'content': content, 'label': label}
            all_data.append(data)

    return all_data


def parse_data_for_gcae(data_path, predict_text=None):
    all_data = []
    if predict_text is not None:
        content = predict_text.strip()
        # label = []
        for idx, name in enumerate(Config.label_names):
            # label.append(0)
            data = {'content': content, 'aspect': Config.label_chinese_name[idx], 'label': 0}
            all_data.append(data)

    else:
        df = pd.read_csv(data_path, encoding='utf-8')
        for index, line in tqdm(df.iterrows()):
            content = line[1].strip()
            # label = []
            for idx, name in enumerate(Config.label_names):
                polarity = map_sentimental_type(int(line[idx+2]))
                label = polarity
                data = {'content': content, 'aspect': Config.label_chinese_name[idx], 'label': label}
                all_data.append(data)

    return all_data


def build_tokenizer(fnames, max_length, data_file, opt):
    if opt.model_name == 'textcnn':
        parse = parse_data_for_textcnn
    elif opt.model_name == 'gcae':
        parse = parse_data_for_gcae

    if os.path.exists(data_file):
        print('loading tokenizer:', data_file)
        tokenizer = pickle.load(open(data_file, 'rb'))
    else:
        tokenizer = Tokenizer.from_files(fnames=fnames, max_length=max_length, parse=parse)
        pickle.dump(tokenizer, open(data_file, 'wb'))
    return tokenizer


class Vocab(object):
    ''' vocabulary of dataset '''
    def __init__(self, vocab_list, add_pad, add_unk):
        self._vocab_dict = dict()
        self._reverse_vocab_dict = dict()
        self._length = 0
        if add_pad: # pad_id should be zero (for Dynamic RNN)
            self.pad_word = '<pad>'
            self.pad_id = self._length      # self.pad_id = 0
            self._length += 1
            self._vocab_dict[self.pad_word] = self.pad_id   # {'<pad>': 0}
        if add_unk:
            self.unk_word = '<unk>'
            self.unk_id = self._length      # self.unk_id = 1
            self._length += 1
            self._vocab_dict[self.unk_word] = self.unk_id   # {'<unk>': 1}
        for w in vocab_list:
            self._vocab_dict[w] = self._length
            self._length += 1
        for w, i in self._vocab_dict.items():   
            self._reverse_vocab_dict[i] = w         # 将索引值作为key
    
    def word_to_id(self, word):     # 查询词的id(索引)
        if hasattr(self, 'unk_id'):
            return self._vocab_dict.get(word, self.unk_id)
        return self._vocab_dict[word]
    
    def id_to_word(self, id_):      # 查询索引对应的词
        if hasattr(self, 'unk_word'):
            return self._reverse_vocab_dict.get(id_, self.unk_word)
        return self._reverse_vocab_dict[id_]
    
    def has_word(self, word):
        return word in self._vocab_dict
    
    def __len__(self):
        return self._length


class Tokenizer(object):
    ''' transform text to indices '''
    def __init__(self, vocab, max_length, lower, pos_char_to_int, pos_int_to_char):
        self.vocab = vocab
        self.max_length = max_length
        self.lower = lower

        self.pos_char_to_int = pos_char_to_int
        self.pos_int_to_char = pos_int_to_char
    
    @classmethod
    def from_files(cls, fnames, max_length, parse, lower=False):
        corpus = set()
        pos_tagged_sent = []
        pos_char_to_int, pos_int_to_char = {}, {}
        for fname in fnames:
            # print(fname)
            for obj in parse(fname):
                text_raw = obj['content']
                if lower:
                    text_raw = text_raw.lower()
                corpus.update(Tokenizer.split_text(text_raw))    # 将文本的所有不重复的词放到 corpus 集合中
        return cls(vocab=Vocab(corpus, add_pad=True, add_unk=True), max_length=max_length, lower=lower, pos_char_to_int=pos_char_to_int, pos_int_to_char=pos_int_to_char)
    
    @staticmethod
    def pad_sequence(sequence, pad_id, maxlen, dtype='int64', padding='post', truncating='post'):
        x = (np.zeros(maxlen) + pad_id).astype(dtype)   # 长度为maxlen的数组中的元素全为pad_id，也就是0
        if truncating == 'pre':
            trunc = sequence[-maxlen:]  # 把过长的句子前面部分截断
        else:
            trunc = sequence[:maxlen]   # 把过长的句子尾部截断
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc      # 在句子尾部打padding
        else:
            x[-len(trunc):] = trunc     # 在句子前面打padding
        return x
    
    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post', max_length=None):
        if self.lower:
            text = text.lower()
        words = Tokenizer.split_text(text)
        sequence = [self.vocab.word_to_id(w) for w in words]    # 一个句子的词索引列表
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence.reverse()      # 将句子索引列表反转
        maxlen = max_length if max_length is not None  else self.max_length
        return Tokenizer.pad_sequence(sequence, pad_id=self.vocab.pad_id, maxlen=maxlen, 
                                      padding=padding, truncating=truncating)
    
    @staticmethod
    def split_text(text):
        # for ch in ["\'s", "\'ve", "n\'t", "\'re", "\'m", "\'d", "\'ll", ",", ".", "!", "*", "/", "?", "(", ")", "\"", "-", ":"]:
        #     text = text.replace(ch, " "+ch+" ")
        # split_mothod = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        split_mothod = lambda x: [y for y in x]  # 以字为单位构建词表
        # return text.strip().split()
        return split_mothod(text.strip())


class SentenceDataset(Dataset):
    ''' PyTorch standard dataset class '''
    def __init__(self, fname, tokenizer, target_dim, opt):
        data = list()

        if opt.model_name == 'textcnn': 
            parse = parse_data_for_textcnn
            for obj in parse(fname, opt.predict_text):
                content = tokenizer.text_to_sequence(obj['content'])
                label = np.asarray(obj['label'], dtype='int64')
                data.append({'content': content, 'label': label})

        elif opt.model_name == 'gcae':
            parse = parse_data_for_gcae
            for obj in parse(fname, opt.predict_text):
                content = tokenizer.text_to_sequence(obj['content'])
                aspect = tokenizer.text_to_sequence(obj['aspect'], max_length=6)
                label = obj['label']
                data.append({'content': content, 'aspect': aspect, 'label': label})

        self._data = data

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __len__(self):
        return len(self._data)


def _load_wordvec(data_path, vocab=None):
    with open(data_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        word_vec = dict()
        for i, line in tqdm(enumerate(f.readlines())):
            tokens = line.strip().split()
            if tokens[0] == '<pad>': # avoid them
                # word_vec['<pad>'] = np.zeros(300, dtype=np.float32)
                continue
            elif tokens[0] == '<unk>':
                word_vec['<unk>'] = np.random.uniform(-0.25, 0.25, 300)
            
            word = ''.join((tokens[:-300]))
            if vocab is None or vocab.has_word(tokens[0]):
                word_vec[word] = np.asarray(tokens[-300:], dtype='float32')

        return word_vec     # {token: vector}

def build_embedding_matrix(vocab, embed_dim, data_file):
    if os.path.exists(data_file):
        print('loading embedding matrix:', data_file)
        embedding_matrix = pickle.load(open(data_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(vocab), embed_dim))
        fname = Config.embedding_path
        word_vec = _load_wordvec(fname, vocab)
        for i in range(len(vocab)):
            vec = word_vec.get(vocab.id_to_word(i))
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(data_file, 'wb'))
    return embedding_matrix