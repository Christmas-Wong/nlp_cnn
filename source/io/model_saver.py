# -*- coding: utf-8 -*-
"""
@Time       : 2021/12/17 8:31 上午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : baseline 
@File       : model_saver.py
@Software   : PyCharm
@Description: 
"""
import os

import torch


def bert_saver(dir_model_save: str, model, tokenizer) -> None:
    """Save Bert Model

    :param dir_model_save: Directory Name
    :param model: Model
    :param tokenizer: Tokenizer
    :return: None
    """

    if not os.path.exists(dir_model_save):
        os.makedirs(dir_model_save)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        model_to_save.state_dict(),
        os.path.join(dir_model_save, "pytorch_model.bin")
    )
    model_to_save.config.to_json_file(
        os.path.join(dir_model_save, "config.json")
    )
    tokenizer.save_vocabulary(
        os.path.join(dir_model_save, "vocab.txt")
    )
