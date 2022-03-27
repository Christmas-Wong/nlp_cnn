# -*- coding: utf-8 -*-
"""
@Time    : 2022/3/19 20:43
@Author  : Fei Wang
@File    : python_runner
@Software: PyCharm
@Description: 
"""
from source.io.reader import yaml_reader
from source.pipeline.classification import run


CONFIG = "E:\\project\\nlp_cnn-main\\data\\config.yml"

if __name__ == "__main__":

    # Read [Config.yml] as dict
    config = yaml_reader(CONFIG)

    # Pass [config] to pipeline program
    run(config)