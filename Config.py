# -*- coding: utf-8 -*-
# @Time    : 2025/1/24 16:37
# @Author  : Gan Liyifan
# @File    : Config.py.py
import torch

class Config:
    def __init__(self):
        self.seed = 123
        self.epochs = 1000
        self.batch_size = 10
        self.max_length = 500
        self.learning_rate = 2e-5
        self.eps = 1e-8
        self.model_name_or_path = 'gpt2'
        self.labels_ids = {'高': 3, '中': 2, '低': 1, '无': 0}
        self.n_labels = len(self.labels_ids)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_data_path = 'data/train'
        self.valid_data_path = 'data/test'

        self.model_save_path = 'checkpoints/model'
        self.tokenizer_save_path = 'checkpoints/tokenizer'
        self.save_step = 100

    def __str__(self):
        return str(self.__dict__)

if __name__ == '__main__':
    config = Config()
    print(config)