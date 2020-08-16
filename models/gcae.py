import torch
import torch.nn as nn
import torch.nn.functional as F


class GCAE(nn.Module):
    # def __init__(self, opt):
    def __init__(self, embedding_matrix, opt):
        super(GCAE, self).__init__()
        self.opt = opt
        
        # V = opt.embed_num
        D = opt.embed_dim
        # C = opt.class_num
        C = opt.polarities_dim
        # A = opt.aspect_num

        Co = opt.num_filters    # 100
        Ks = opt.filter_sizes   # (3, 4, 5)

        # self.embed = nn.Embedding(V, D)
        # self.embed.weight = nn.Parameter(opt.embedding, requires_grad=True)
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)

        # self.aspect_embed = nn.Embedding(A, opt.aspect_embed_dim)
        # self.aspect_embed.weight = nn.Parameter(opt.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])


        # self.convs3 = nn.Conv1d(D, 300, 3, padding=1), smaller is better
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(100, Co)


    # def forward(self, text, aspect):
    def forward(self, inputs):
        text, aspect = inputs[0], inputs[1]
        text = self.embed(text)         # torch.Size([64, 500, 300])
        aspect_v = self.embed(aspect)   # torch.Size([64, 6, 300])

        aa = [F.relu(conv(aspect_v.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)
        # aa = F.tanhshrink(self.convs3(aspect_v.transpose(1, 2)))  # [(N,Co,L), ...]*len(Ks)
        # aa = F.max_pool1d(aa, aa.size(2)).squeeze(2)
        # aspect_v = aa
        # smaller is better

        # x = [F.tanh(conv(text.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        # y = [F.relu(conv(text.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [torch.tanh(conv(text.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(text.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x = [F.adaptive_max_pool1d(i, 2) for i in x]
        # x = [i.view(i.size(0), -1) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        logit = self.fc1(x)  # (N,C)

        # return logit, x, y
        return logit
