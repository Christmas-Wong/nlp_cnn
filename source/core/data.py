# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/19 11:18 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : data.py
@Software   : PyCharm
@Description: 
"""
import jieba
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from ..io.reader import read_single_file
from ..embedding.word2vec import word2vec_train


class MyDataset(Dataset):
    def __init__(self, text: list, label: list):
        self.label = label
        self.origin_text = text
        self.text = None
        self.corpus = list()

        for sentence in self.origin_text:
            words = jieba.lcut(sentence)
            self.corpus += words

    def __getitem__(self, item):
        return self.text[item], self.label[item]

    def __len__(self):
        return len(self.label)

    def get_corpus(self):
        return self.corpus

    def get_origin_text(self):
        return self.origin_text


class Vocab(object):
    def __init__(self, *datasets: MyDataset):
        super(Vocab, self).__init__()
        self.embedding = None
        self.corpus = list()
        self.words = list()
        for dataset in datasets:
            self.corpus += dataset.get_origin_text()
            self.words += dataset.get_corpus()
        self.id_2_word = dict()
        self.word_2_id = dict()
        self.vocab = [w for w, v in Counter(self.words).most_common() if v > 1]
        self.vocab = ["unk", "pad"] + self.vocab
        for index, word in enumerate(self.vocab):
            self.id_2_word[index] = word
            self.word_2_id[word] = index

    def get_embedding(self, embedding_type: str = "word2vec", embedding_dim: int = 128):
        self.embedding = np.zeros([len(self.vocab), embedding_dim])
        if "word2vec" == embedding_type:
            self.__word2vec_embedding(embedding_dim)
            return self.embedding

    def __word2vec_embedding(self, embedding_dim: int):
        self.vectors = word2vec_train(self.corpus, embedding_dim=embedding_dim)
        for word, index in tqdm(self.word_2_id.items()):
            if word in self.vectors.wv.key_to_index.keys():
                self.embedding[index, :] = self.vectors.wv[word]
            elif word == "pad":
                self.embedding[index, :] = np.zeros(embedding_dim)
            else:
                self.embedding[index, :] = 0.2 * np.random.random(embedding_dim) - 0.1


def load_dataset(data_files: dict, file_extension: str) -> dict:
    result = dict()
    for key, value in data_files.items():
        json_list = read_single_file(value, file_extension)
        result[key] = MyDataset(
            [ele["text"] for ele in json_list],
            [ele["label"] for ele in json_list]
        )
    return result


def convert_sentence(sentence: str, word_2_id: dict, max_length: int):
    unk_id = word_2_id["unk"]
    pad_id = word_2_id["pad"]

    ids = [word_2_id.get(word, unk_id) for word in jieba.lcut(sentence)]
    if len(ids) >= max_length:
        ids = ids[:max_length]
    else:
        ids += [pad_id] * (max_length - len(ids))
    return ids


