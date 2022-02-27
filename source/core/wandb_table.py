# -*- coding: utf-8 -*-
"""
@Time       : 2021/12/17 1:58 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : ysh 
@File       : wandb_table.py
@Software   : PyCharm
@Description: 
"""
import wandb
from typing import List


def wandb_confusion(list_true: List, list_pred: List, labels: List):
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=list_true,
                preds=list_pred,
                class_names=labels
            )
        }
    )


def wandb_pr_recall_f1(evaluation_index: str, report: dict, labels: List):
    data_report = [[ele, report[ele][evaluation_index]] for ele in labels]
    table = wandb.Table(
        data=data_report,
        columns=["class_name", evaluation_index]
    )
    wandb.log(
        {
            evaluation_index+"_chart": wandb.plot.bar(
                table,
                "class_name",
                evaluation_index,
                title="Per Class "+evaluation_index
            )
        }
    )