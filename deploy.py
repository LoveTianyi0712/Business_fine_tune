# -*- coding: utf-8 -*-
# @Time    : 2025/2/3 12:23
# @Author  : Gan Liyifan
# @File    : deploy.py
import os
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Config import Config
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from data.dataloader import BusinessIntention, BusinessIntentionDataset
from train import validation_epoch
from utils.utils import Gpt2ClassificationCollator


def deploy_model(config):
    folders = [f for f in os.listdir(config.model_save_path) if
               os.path.isdir(os.path.join(config.model_save_path, f))]

    epoch_numbers = []
    for folder in folders:
        match = re.match(r'epoch_(\d+)', folder)
        if match:
            epoch_numbers.append(int(match.group(1)))

    if not epoch_numbers:
        raise ValueError("No valid checkpoints found in the model_save_path")

    max_epoch = max(epoch_numbers)
    print("Loading model from epoch {}".format(max_epoch))
    model_path = os.path.join(config.model_save_path, "epoch_{}".format(max_epoch))
    tokenizer_path = os.path.join(config.tokenizer_save_path, "epoch_{}".format(max_epoch))

    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path,
                                              num_labels=config.n_labels)

    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=tokenizer_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path,
                                                          config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(config.device)
    print('Model loaded to `%s`' % config.device)

    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=config.labels_ids,
                                                              max_sequence_len=config.max_length)

    # load data
    id = 'XSJ20241129040337'
    grade = '四年级'
    project = '素养'
    description = '，，四年级，，考虑英语嗯课程，，地铁大厦校区，，林明坤妈妈推荐，，小宝'
    channel = '他人推荐'
    student_type = '未知'

    business_intention = BusinessIntention(id, grade, project, description, channel, student_type)
    data_list = [business_intention]
    dataset = BusinessIntentionDataset()
    dataset.load_data_from_class(data_list)

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=gpt2_classification_collator)

    predictions_labels = []
    total_loss = 0
    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.type(torch.long).to(config.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    print(predictions_labels, total_loss)


if __name__ == '__main__':
    config = Config()
    print(config)
    deploy_model(config)