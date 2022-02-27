# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/20 5:06 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : main.py.py
@Software   : PyCharm
@Description:

    Program Entry and Pass Configurations to pipeline program

"""
import argparse
from source.io.reader import yaml_reader
from source.pipeline.classification import run


if __name__ == "__main__":

    # Get [config] parameter in [run.sh] file
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    # Read [Config.yml] as dict
    config = yaml_reader(args.config)

    # Pass [config] to pipeline program
    run(config)
