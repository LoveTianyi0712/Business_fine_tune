# -*- coding: utf-8 -*-
# @Time    : 2025/1/25 10:27
# @Author  : Gan Liyifan
# @File    : dataloader.py
import os
import json
from torch.utils.data import Dataset
from tqdm import tqdm


class BusinessIntentionDataset(Dataset):
    def __init__(self, path):
        # Check if path exists.
        if not os.path.isdir(path):
            raise ValueError('Invalid `path` variable! Needs to be a directory')

        self.data = []

        # Get all JSON files from path.
        files_names = [f for f in os.listdir(path) if f.endswith('.json')]

        # Go through each file and read its content.
        for file_name in tqdm(files_names, desc='Loading JSON files'):
            file_path = os.path.join(path, file_name)

            # Read content.
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        content = json.loads(line)
                        # Create a dictionary with id, text, and label
                        data_entry = {
                            'id': content['id'],
                            'text': ' '.join([f"{key}: {value}" for key, value in content.items() if key not in ['id', 'intention']]),
                            'label': content['intention']
                        }
                        self.data.append(data_entry)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'id': self.data[idx]['id'],
            'text': self.data[idx]['text'],
            'label': self.data[idx]['label']
        }


if __name__ == '__main__':
    dataset = BusinessIntentionDataset('.')
    print(dataset[0])
