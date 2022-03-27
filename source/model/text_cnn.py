# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/19 5:26 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : template.py
@Software   : PyCharm
@Description: 
"""
import torch
from torch import nn
import torch.nn.functional as F
from ..core.args import TextCNNArguments


class TextCNN(nn.Module):
    def __init__(self, config: TextCNNArguments):
        super().__init__()

        # 定义embedding层：随机初始化embedding层还是使用训练好的预训练向量
        # if config.embedding_type == "random", 随机初始化
        # if config.embedding_type != "random", 使用预训练好的向量
        if "random" != config.embedding_type:
            self.embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_dim
            ).from_pretrained(config.embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(
                config.vocab_size,
                config.embedding_dim
            )
        # 定义卷积层：输入channel为1, 输出channel为卷积核的个数
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=config.num_filters[index],
                      kernel_size=(fs, config.embedding_dim))
            for index, fs in enumerate(config.filters)
        ])

        feature_dim = sum(config.num_filters)
        # 定义全连接层，输出为标签个数
        self.fc = nn.Linear(feature_dim, config.num_labels)
        # 定义dropout, 参数为dropout的概率，默认为0.5
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # text = [batch size, sent len]

        embedded = self.embedding(inputs)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        output = self.fc(cat)

        return output
