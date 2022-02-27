# -*- coding: utf-8 -*-
"""
@Time       : 2022/1/19 8:56 下午
@Author     : Wang Fei
@Last Editor: Wang Fei
@Project    : template
@File       : classification.py
@Software   : PyCharm
@Description: 
"""
import os
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from ..core.args import TrainingArguments, TextCNNArguments
from ..core.data import load_dataset, Vocab, convert_sentence
from ..io.writer import json_writer, write_lines
from ..model.text_cnn import TextCNN
from ..core.trainer import MyTrainer
from ..core.util import log_init
from ..core.inference import Inference
from ..io.reader import read_single_file, list_from_txt
from ..io.writer import report_saver, confusion_2_csv


def run(config: dict):
    # Initial Loguru
    log_init(config["log"])
    # Initial Wandb
    wandb.init(
        project=config["project_info"]["project_name"],
        group=config["project_info"]["group_name"],
        name=config["project_info"]["run_name"],
    )

    training_args = TrainingArguments(**config["train_arguments"])
    text_cnn_args = TextCNNArguments(**config["text_cnn"])
    data_files = {
        "train": training_args.train_file,
        "test": training_args.test_file,
        "eval": training_args.eval_file
    }
    raw_dataset = load_dataset(file_extension="json", data_files=data_files)
    vocab = Vocab(raw_dataset["train"], raw_dataset["test"], raw_dataset["eval"])

    if "random" != text_cnn_args.embedding_type:
        embedding = vocab.get_embedding(embedding_type=text_cnn_args.embedding_type,
                                        embedding_dim=text_cnn_args.embedding_dim)
        text_cnn_args.embedding = torch.FloatTensor(embedding)

    label_list = list(set(raw_dataset["train"].label))
    label_list.sort()

    # label_2_id
    label_2_id = {}
    id_2_label = {}
    for index, ele in enumerate(label_list):
        label_2_id[ele] = index
        id_2_label[index] = ele

    text_cnn_args.vocab_size = len(vocab.vocab)
    text_cnn_args.num_labels = len(label_list)
    model = TextCNN(text_cnn_args)

    for key, value in raw_dataset.items():
        value.text = [torch.tensor(convert_sentence(sentence, vocab.word_2_id, max_length=config["padding"])) for sentence in
                      value.origin_text]
        value.label = [torch.tensor(label_2_id[ele]) for ele in value.label]

    optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = MyTrainer(
        model,
        criterion,
        optimizer,
        DataLoader(raw_dataset["train"], batch_size=training_args.train_batch_size, shuffle=True),
        DataLoader(raw_dataset["eval"], batch_size=training_args.eval_batch_size, shuffle=True),
        training_args
    )
    trainer.train()

    json_writer(
        label_2_id,
        os.path.join(
            training_args.dir_output,
            "label_2_id.json"
        )
    )
    json_writer(
        id_2_label,
        os.path.join(
            training_args.dir_output,
            "id_2_label.json"
        )
    )
    write_lines(
        vocab.vocab,
        os.path.join(
            training_args.dir_output,
            "vocab.txt"
        )
    )

    vocab_list = list_from_txt(
        os.path.join(
            training_args.dir_output,
            "vocab.txt"
        )
    )
    vocab_dict = dict()
    for index, ele in enumerate(vocab_list):
        vocab_dict[ele] = index
    model_ckpt = torch.load(
        os.path.join(
            training_args.dir_output,
            "pytorch_text_cnn.pt"
        )
    )
    best_model = TextCNN(text_cnn_args)
    best_model.load_state_dict(model_ckpt['net'])
    predict_data = read_single_file(training_args.test_file, "json")
    inference = Inference(
        best_model,
        predict_data,
        label_2_id,
        id_2_label,
        vocab_dict,
        config["padding"],
        True
    )
    report_table, report_txt, confusion = inference.run()
    report_saver(
        report_table,
        report_txt,
        training_args.dir_output
    )
    confusion_2_csv(
        confusion,
        list(label_2_id.keys()),
        os.path.join(
            training_args.dir_output,
            "confusion_matrix.csv"
        )
    )
    json_writer(
        predict_data,
        os.path.join(
            training_args.dir_output,
            "predict.json"
        )
    )
