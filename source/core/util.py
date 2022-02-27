# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/4 2:23 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : ysh
@File       : util.py
@Software   : PyCharm
@Description:
"""
import os
import time
import torch
import numpy as np
from typing import List
from loguru import logger
from functools import wraps
from pandas import DataFrame


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     # 这里会存储迄今最优模型的参数
        torch.save(model, 'early_stop_model.pt')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss


def log_init(config: dict):
    logger.add(
        config["file"],
        rotation=config["rotation"],
        compression=config["compression"]
    )


def directory_exist(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_str_time(pattern: str = "%Y_%m_%d %H:%M:%S"):
    result = str(time.strftime(pattern, time.localtime()))
    return result


def calculate_time(place=2):
    """
    计算函数运行时间装饰器

    :param place: 显示秒的位数，默认为2位
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            beg = time.time()
            f = func(*args, **kwargs)
            end = time.time()
            s = '{}：{:.%sf} s' % place
            logger.info(s.format("函数处理耗时", end - beg))
            return f
        return wrapper
    return decorator


def list_to_json(list_text: List, list_label: List) -> List:

    result = []
    for text, label in zip(list_text, list_label):
        ele = {
            "text": text,
            "label": label
        }
        result.append(ele)
    return result


def json_2_df(json_data: List) -> DataFrame:
    columns = list(json_data[0].keys())
    list_data = [[] for _ in columns]
    for ele in json_data:
        for key, value in ele.items():
            index = columns.index(key)
            list_data[index].append(value)

    result = DataFrame()
    for index, ele in enumerate(columns):
        result[ele] = list_data[index]
    return result


def df_2_json(df: DataFrame) -> List:
    columns = df.columns
    result = list()
    for index, row in df.iterrows():
        dict_ele = dict()
        for name in columns:
            dict_ele[name] = row[name]
        result.append(dict_ele)
    return result