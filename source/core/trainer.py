# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/19 7:08 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : trainer.py
@Software   : PyCharm
@Description: 
"""
import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn import metrics
from .util import EarlyStopping
from ..core.args import TrainingArguments
from torch.optim.lr_scheduler import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class MyTrainer(object):
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 train_dataset,
                 eval_dataset,
                 config: TrainingArguments
                 ) -> None:
        super(MyTrainer, self).__init__()
        self.model = model.to(DEVICE)
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.config = config
        self.early_stopping = EarlyStopping(config.early_stopping, verbose=True)
        self.best_score = float('inf')
        self.dataset_train = train_dataset
        self.dataset_eval = eval_dataset
        self.scheduler = StepLR(self.optimizer, step_size=5)

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.config.num_train_epoch)):
            loss_train = 0
            for batch_x, batch_y in tqdm(self.dataset_train):
                torch.cuda.empty_cache()
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                # 每个batch都以独立训练，因此每训练一个batch，都需要将梯度设置为零
                self.optimizer.zero_grad()
                logits = self.model(batch_x)
                loss_ele = self.criterion(logits, batch_y)
                # 反向传播
                loss_ele.backward()
                # 优化器参数更新
                self.optimizer.step()
                self.step += 1
                loss_train += loss_ele
                if self.config.eval_step > 0 and self.step % self.config.eval_step == 0:
                    fscore, loss_eval = self.__eval()
                    wandb.log({"Train Loss": loss_eval, "Eval Loss": loss_eval, "Eval fscore": fscore, "epoch": epoch})
            loss_train /= len(self.dataset_train)
            logger.info('Train loss : {:.6f}\n'.format(loss_train))
            # Eval model score
            if self.config.eval_step == 0:
                fscore, loss_eval = self.__eval()
                wandb.log({"Train Loss": loss_eval, "Eval Loss": loss_eval, "Eval fscore": fscore, "epoch": epoch})
            # Update Model
            if loss_train <= self.best_score:
                self.best_score = loss_train
                self.save_model(epoch)
            self.early_stopping(loss_train, self.model)
            if self.early_stopping.early_stop:
                logger.info("Early stopping \n")
                break
            self.scheduler.step()

    def __eval(self):
        logger.info("Eval \n")
        self.model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        label_all = np.array([], dtype=int)
        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.dataset_train):
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                loss_total += loss
                predict = torch.max(logits, 1)[1].cpu().numpy()
                label = batch_y.data.cpu().numpy()
                label_all = np.append(label_all, label)
                predict_all = np.append(predict_all, predict)
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(label_all, predict_all)
        loss = loss_total / len(self.dataset_eval)
        logger.info('Evaluation \n Loss : 【{}】, precision :【{}】, recall : 【{}】, fscore : 【{}】 \n'.format(loss,
                                                                                                         np.mean(precision),
                                                                                                         np.mean(recall),
                                                                                                         np.mean(fscore)))
        return np.mean(fscore), loss

    def save_model(self, epoch: int):
        state = {
            'epoch': epoch,
            'net': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(
            state,
            os.path.join(self.config.dir_output, "pytorch_text_cnn.pt")
        )
