# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/19 17:32
@Author  : Fei Wang
@File    : dpcnn
@Software: PyCharm
@Description: 
"""

import torch
import torch.nn as nn
from ..core.args import DPCNNArguments

torch.manual_seed(1)


class DPCNN(nn.Module):
    def __init__(self, config: DPCNNArguments):
        super(DPCNN, self).__init__()
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
        self.conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_dim), stride=1)
        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_labels)
        # 定义dropout, 参数为dropout的概率，默认为0.5
        self.dropout = nn.Dropout(config.dropout)
        self.bn_1 = nn.BatchNorm2d(num_features=config.num_filters)
        self.bn_2 = nn.BatchNorm2d(num_features=config.num_filters)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv_region(x)  # [batch_size, 250, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        x = self.padding1(x)  # [batch_size, 250, seq_len, 1]
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, 250, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = nn.functional.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = nn.functional.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

