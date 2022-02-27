# -*- coding: utf-8 -*-
"""
@Time       : 2021/12/16 7:25 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : baseline 
@File       : writer.py
@Software   : PyCharm
@Description: 
"""
import os
import json
import pickle
import codecs
import pandas as pd
from typing import List
from loguru import logger
from prettytable import PrettyTable
from sklearn.preprocessing import LabelEncoder


def write_lines(content: List, file: str) -> None:
    """Write str into File line by line

    :param content: Str inputs
    :param file: Target File
    :return: None
    """
    with codecs.open(file, "w", encoding="utf_8") as f:
        for ele in content:
            f.writelines(ele)
            f.write("\n")
    f.close()
    logger.info("Write into [{}] [{}]lines Successfully.".format(file, len(content)))


def report_saver(report_dict: PrettyTable, report_txt: str, dir_path: str) -> None:
    """Save PrettyTable_report as CSV File or STR _Report into TXT_File

    :param report_dict: PrettyTable Report
    :param report_txt: STR_Report
    :param dir_path: Target Dir
    :return: None
    """
    if report_dict:
        file = report_dict.get_csv_string()
        base_writer(
            file,
            os.path.join(dir_path, "report.csv")
        )
        logger.info("Save report.csv Successfully.")
    if report_txt:
        base_writer(
            report_txt,
            os.path.join(dir_path, "report.txt")
        )
        logger.info("Save report.txt Successfully.")


def label_encoder_saver(le: LabelEncoder, file: str) -> None:
    """Save LabelEncoder into File

    :param le: LabelEncoder
    :param file: Target File
    :return: None
    """
    with codecs.open(file, "wb") as f:
        pickle.dump(le, f)
    f.close()
    logger.info("Save LabelEncoder Successfully.")


def base_writer(txt: str, path: str) -> None:
    """Write str

    Args:
        txt (str): Content to be written,
        path (str): Path of File

    Returns:
        None
    """

    # if file exist, delete it
    if os.path.exists(path):
        os.remove(path)

    with codecs.open(path, "w") as f:
        f.write(txt)
    f.close()
    logger.info("Base Writer Work Successfully.")


def confusion_2_csv(matrix: List, labels: List, file: str) -> None:
    """Save Confusion Matrix into CSV File

    :param matrix: Confusion Matrix
    :param labels: Labels
    :param file: Target File
    :return: None
    """
    df_confusion = pd.DataFrame(matrix)
    df_confusion.columns = labels
    df_confusion["labels"] = labels
    df_confusion.set_index(["labels"], inplace=True)
    df_confusion.to_csv(file, encoding="utf_8_sig")
    logger.info("Save Confusion Matrix Successfully.")


def json_writer(json_object, file: str) -> None:
    """Save Json_List into File as dict format

    :param json_object: List of Json
    :param file: Target File
    :return: None
    """
    with open(file, 'w') as f:
        json.dump(
            json_object,
            f,
            indent=4,
            ensure_ascii=False
        )
    f.close()
    logger.info("Save Json Successfully.")


def jsonl_writer(content: List, file: str) -> None:
    """ Write json_list as jsonl into file

    :param content: List of json
    :param file: Target File
    :return: None
    """
    with codecs.open(file, "w") as f:
        for json_object in content:
            json_str = json.dumps(json_object, ensure_ascii=False)
            f.writelines(json_str)
            f.write("\n")
    f.close()
    logger.info("Write [{}]lines Successfully.".format(len(content)))

