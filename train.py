# -*- coding: utf-8 -*-
# @Time    : 2025/1/26 9:31
# @Author  : Gan Liyifan
# @File    : train.py
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)

from Config import Config
from data.dataloader import BusinessIntentionDataset
from utils.utils import Gpt2ClassificationCollator


def train_epoch(model, dataloader, optimizer_, scheduler_, device_):
    predictions_labels = []
    true_labels = []

    total_loss = 0
    model.train()
    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}
        model.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_.step()
        scheduler_.step()
        logits = logits.detach().cpu().numpy()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


def validation_epoch(model, dataloader, device_):
    predictions_labels = []
    true_labels = []

    total_loss = 0
    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            total_loss += loss.item()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss


def train(config):
    print("Loading configuration")
    print(config)
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                              num_labels=config.n_labels)
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    print('Loading model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=config.model_name_or_path,
                                                          config=model_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model.to(config.device)
    print('Model loaded to `%s`' % config.device)

    gpt2_classification_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                              labels_encoder=config.labels_ids,
                                                              max_sequence_len=config.max_length)

    dataset = BusinessIntentionDataset(config.train_data_path)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                  collate_fn=gpt2_classification_collator)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                                  collate_fn=gpt2_classification_collator)

    # valid_dataset = BusinessIntentionDataset(config.valid_data_path)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
    #                               collate_fn=gpt2_classification_collator)

    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate,  # default is 5e-5, our notebook had 2e-5
                      eps=config.eps  # default is 1e-8.
                      )

    total_steps = len(train_dataloader) * config.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    all_loss = {'train_loss': [], 'val_loss': []}
    all_acc = {'train_acc': [], 'val_acc': []}

    for epoch in range(config.epochs):
        print('Epoch: {}'.format(epoch))
        print('Training on batches...')
        train_labels, train_predict, train_loss = train_epoch(model, train_dataloader, optimizer, scheduler,
                                                              config.device)
        train_acc = accuracy_score(train_labels, train_predict)

        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation_epoch(model, valid_dataloader, config.device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        print("train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" % (
            train_loss, val_loss, train_acc, val_acc))
        print()

        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)

        if epoch % config.save_step == 0:
            model.save_pretrained(config.model_save_path)
            tokenizer.save_pretrained(config.tokenizer_save_path)


if __name__ == '__main__':
    config = Config()
    print(config)
    train(config)
