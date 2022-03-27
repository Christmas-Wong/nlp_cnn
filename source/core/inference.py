# -*- coding: utf-8 -*-
"""
@Time       : 2022/2/7 11:39 AM
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : text_cnn 
@File       : inference.py
@Software   : PyCharm
@Description: 
"""
import torch
from tqdm import tqdm
from loguru import logger
from prettytable import PrettyTable
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from ..core.data import convert_sentence
from ..core.wandb_table import wandb_confusion, wandb_pr_recall_f1
from torch.utils.data import DataLoader


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pretty_report(report: dict):
    result = PrettyTable()
    result.field_names = ["Class", "precision", "recall", "f1-score", "support"]
    for key, value in report.items():
        if key == "accuracy":
            result.add_row([key, 0, 0, value, report["weighted avg"]["support"]])
            continue
        result.add_row([key, str(value["precision"]), str(value["recall"]), str(value["f1-score"]), str(value["support"])])
    return result


class Inference(object):
    def __init__(self,
                 model,
                 data: DataLoader,
                 label_2_id: dict,
                 id_2_label: dict,
                 vocab: dict,
                 padding: int,
                 evaluate_status: bool = False
                 ):
        super(Inference, self).__init__()
        self.model = model.to(DEVICE)
        self.data = data
        self.label_2_id = label_2_id
        self.id_2_label = id_2_label
        self.vocab = vocab
        self.evaluate_status = evaluate_status
        self.padding = padding

    def run(self):
        self.model.eval()
        with torch.no_grad():
            for ele in tqdm(self.data):
                sentence = torch.tensor([convert_sentence(ele["text"], self.vocab, max_length=self.padding)]).to(DEVICE)
                outputs = self.model(sentence)
                if list(outputs.size())[0] == 1:
                    predict = torch.max(outputs.data, 1)[1].cpu().item()
                else:
                    predict = torch.max(outputs.data, 0)[1].cpu().item()
                ele["predict"] = self.id_2_label[predict]
        if self.evaluate_status:
            return self.evaluate()
        return None, None, None

    def evaluate(self):
        list_true = [self.label_2_id[ele["label"]] for ele in self.data]
        list_pred = [self.label_2_id[ele["predict"]] for ele in self.data]

        confusion = confusion_matrix(list_true, list_pred)
        logger.info("\n" + str(confusion))

        report_dict = classification_report(
            list_true,
            list_pred,
            target_names=self.label_2_id.keys(),
            output_dict=True,
            digits=4)
        report_txt = classification_report(
            list_true,
            list_pred,
            target_names=self.label_2_id.keys(),
            output_dict=False,
            digits=4)
        report_table = pretty_report(report_dict)
        logger.info("\n" + str(report_table))
        wandb_confusion(list_true, list_pred, list(self.label_2_id.keys()))
        for ele in ["precision", "recall", "f1-score"]:
            wandb_pr_recall_f1(ele, report_dict, list(self.label_2_id.keys()))

        return report_table, report_txt, confusion

