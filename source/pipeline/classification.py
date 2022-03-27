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
from ..core.args import TrainingArguments, TextCNNArguments, DPCNNArguments
from ..core.data import load_dataset, Vocab, convert_sentence
from ..io.writer import json_writer, write_lines
from ..model.text_cnn import TextCNN
from ..model.dpcnn import DPCNN
from ..core.trainer import MyTrainer
from ..core.util import log_init, directory_exist
from ..core.inference import Inference
from ..io.reader import read_single_file, list_from_txt
from ..io.writer import report_saver, confusion_2_csv

dict_model = {
    "text_cnn": TextCNN,
    "dpcnn": DPCNN,
}
dict_args = {
    "text_cnn": TextCNNArguments,
    "dpcnn": DPCNNArguments,
}
# DIRs that save different results
PROJECT_DIRS = ["model", "cache", "checkpoints"]


def run(config: dict):
    model_name = config["project_info"]["model"]
    # DIR CHECK
    output_dir = os.path.join(
        config["train_arguments"]["dir_output"],
        "_".join(config["project_info"].values())
    )
    for dir_ele in PROJECT_DIRS:
        directory_exist(
            os.path.join(
                output_dir,
                dir_ele
            )
        )
    # log initial
    config["log"]["file"] = os.path.join(
        output_dir,
        "cache",
        "runtime.log"
    )
    log_init(config["log"])
    # Initial Wandb
    wandb.init(
        project=config["project_info"]["project_name"],
        group=config["project_info"]["group_name"],
        name=config["project_info"]["run_name"],
    )

    training_args = TrainingArguments(**config["train_arguments"])
    model_args = dict_args[model_name](**config[model_name])
    data_files = {
        "train": training_args.train_file,
        "test": training_args.test_file,
        "eval": training_args.eval_file
    }
    raw_dataset = load_dataset(file_extension="json", data_files=data_files)
    vocab = Vocab(raw_dataset["train"], raw_dataset["test"], raw_dataset["eval"])

    if "random" != model_args.embedding_type:
        embedding = vocab.get_embedding(embedding_type=model_args.embedding_type,
                                        embedding_dim=model_args.embedding_dim)
        model_args.embedding = torch.FloatTensor(embedding)

    label_list = list(set(raw_dataset["train"].label))
    label_list.sort()

    # label_2_id
    label_2_id = {}
    id_2_label = {}
    for index, ele in enumerate(label_list):
        label_2_id[ele] = index
        id_2_label[index] = ele

    model_args.vocab_size = len(vocab.vocab)
    model_args.num_labels = len(label_list)
    model = dict_model[model_name](model_args)

    for key, value in raw_dataset.items():
        value.text = [torch.tensor(convert_sentence(sentence, vocab.word_2_id, max_length=config["padding"])) for sentence in
                      value.origin_text]
        value.label = [torch.tensor(label_2_id[ele]) for ele in value.label]

    optimizer = optim.Adam(model.parameters(), lr=training_args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    training_args.dir_output = os.path.join(
        output_dir,
        "checkpoints",
    )
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
            output_dir,
            "cache",
            "id_2_label.json"
        )
    )
    write_lines(
        vocab.vocab,
        os.path.join(
            output_dir,
            "cache",
            "vocab.txt"
        )
    )

    vocab_list = list_from_txt(
        os.path.join(
            output_dir,
            "cache",
            "vocab.txt"
        )
    )
    vocab_dict = dict()
    for index, ele in enumerate(vocab_list):
        vocab_dict[ele] = index
    model_ckpt = torch.load(
        os.path.join(
            output_dir,
            "checkpoints",
            "pytorch_model.pt"
        )
    )

    best_model = dict_model[config["project_info"]["model"]](model_args)
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
        os.path.join(
            output_dir,
            "cache",
        )
    )
    confusion_2_csv(
        confusion,
        list(label_2_id.keys()),
        os.path.join(
            output_dir,
            "cache",
            "confusion_matrix.csv"
        )
    )
    json_writer(
        predict_data,
        os.path.join(
            output_dir,
            "cache",
            "predict.json"
        )
    )
