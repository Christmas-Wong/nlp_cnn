# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/19 8:31 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : word2vec.py
@Software   : PyCharm
@Description: 
"""
import os
import jieba
import time
from tqdm import tqdm
from loguru import logger
import multiprocessing
from gensim.models import Word2Vec


def word2vec_train(corpus: list, dir_output: str = None, embedding_dim: int = 128) -> Word2Vec:
    vocab = list()
    word_list = list()
    for sentence in tqdm(corpus):
        words = jieba.lcut(sentence)
        word_list.append(words)
        vocab += words

    logger.info("Word2vec Training Start \n")
    start_time = time.time()
    model = Word2Vec(
        word_list,
        min_count=2,
        vector_size=embedding_dim,
        workers=multiprocessing.cpu_count(),
        window=8,
        sg=0,
        negative=5,
        ns_exponent=0.75,
        cbow_mean=1,
        hashfxn=hash,
        hs=0,
        epochs=100
    )
    end_time = time.time()
    logger.info("Word2vec Training End, Cost【{}】Seconds \n".format(end_time-start_time))
    if dir_output:
        model.wv.save_word2vec_format(
            os.path.join(
                dir_output,
                "word2vec.bin"
            ),
            binary=False
        )
    return model