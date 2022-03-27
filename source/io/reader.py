# -*- coding: utf-8 -*-
"""
@Time       : 2021/12/16 4:34 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : project
@File       : reader.py
@Software   : PyCharm
@Description: 
"""
import yaml
import json
import codecs
import pickle
import pandas as pd
from typing import List
from loguru import logger
from sklearn.preprocessing import LabelEncoder


def read_single_file(file: str, file_extension: str):
    if "json" == file_extension:
        return json_reader(file)
    elif "jsonl" == file_extension:
        return jsonl_reader(file)
    elif "csv" == file_extension:
        df = pd.read_csv(
            file,
            encoding="utf_8"
        )
        result = list()
        for index, row in df.items():
            ele = {
                "text": str(row["text"]).strip(),
                "label": row["label"]
            }
            result.append(ele)
        return result
    else:
        raise Exception("[{}] file extension unknown!".format(file_extension))


def label_encoder_reader(file: str) -> LabelEncoder:
    """Get LabelEncoder from File

    :param file: File
    :return: LabelEncoder
    """
    with codecs.open(file, "rb", encoding="utf_8") as f:
        result = pickle.load(f)
    f.close()
    return result


def yaml_reader(file: str) -> dict:
    """Get Json from yaml

    :param file: YAML File
    :return: dict
    """
    with codecs.open(file, "r", encoding="utf_8") as f:
        content = f.read()
        result = yaml.safe_load(content)
    return result


def jsonl_reader(file: str) -> List:
    """Get Json_List from jsonl File

    :param file: JSONL file
    :return:
    """
    result = list()
    with codecs.open(file, "r", encoding="utf_8") as f:
        for line in f.readlines():
            json_ele = json.loads(line)
            result.append(json_ele)
    f.close()
    logger.info("Read [{}] lines from JSONL File".format(len(result)))
    return result


def json_reader(file: str):
    """Get Json

    :param file: JSON File
    :return:
    """
    with codecs.open(file, "r", encoding="utf_8") as f:
        result = json.load(f)
    f.close()
    return result


def list_from_txt(file: str) -> List:
    """Get STR_List from File

    :param file: TXT File
    :return:
    """
    result = []
    with codecs.open(file, "r", encoding="utf_8") as f:
        for line in f.readlines():
            content = line.strip()
            result.append(content)
    logger.info("\n Read [{}] lines from file".format(len(result)))
    return result
