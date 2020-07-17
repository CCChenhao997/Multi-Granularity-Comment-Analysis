import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(TextCNN, self).__init__()
        self.num_classes = opt.num_classes
        self.polarities_dim = opt.polarities_dim

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
            
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opt.num_filters, (k, opt.embed_dim)) for k in opt.filter_sizes])   # * num_filters卷积核数量(channels)=256, filter_sizes = (2, 3, 4) 卷积核尺寸
        
        self.dropout = nn.Dropout(opt.dropout)
        self.fc = nn.Linear(opt.num_filters * len(opt.filter_sizes), opt.num_classes * opt.polarities_dim)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, inputs):
        text = inputs[0]
        out = self.embed(text)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)   # * 按列拼接
        out = self.dropout(out)
        out = self.fc(out)                                           # [batch size, num_classes * num_labels]
        out = out.view((-1, self.num_classes, self.polarities_dim))  # [batch size, num_classes, num_labels]
        return out