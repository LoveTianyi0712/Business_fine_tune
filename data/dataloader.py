# -*- coding: utf-8 -*-
# @Time    : 2025/1/25 10:27
# @Author  : Gan Liyifan
# @File    : dataloader.py
import os
import json
from torch.utils.data import Dataset
from tqdm import tqdm


class BusinessIntentionDataset(Dataset):
    def __init__(self):
        self.data = []

    def load_data_from_path(self, path):
        if not os.path.isdir(path):
            raise ValueError('Invalid `path` variable! Needs to be a directory')

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
                            'text': ' '.join([f"{key}: {value}" for key, value in content.items() if
                                              key not in ['id', 'intention']]),
                            'label': content['intention']
                        }
                        self.data.append(data_entry)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file_name}: {e}")

    def load_data_from_class(self, data_list):
        if not(isinstance(data_list, list)):
            raise TypeError("data_list must be a list of BusinessIntention objects")

        for data in data_list:
            self.data.append(data.data_entry())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'id': self.data[idx]['id'],
            'text': self.data[idx]['text'],
            'label': self.data[idx]['label']
        }


class BusinessIntention:
    def __init__(self, id, grade, project, description, channel, student_type, intention=None):
        self.grade = grade
        self.project = project
        self.description = description
        self.channel = channel
        self.student_type = student_type
        self.intention = intention

        self.id = id
        self.text = self.data_to_text()
        self.label = self.intention

    def data_to_text(self):
        data_dict = {
            'id': self.id,
            'grade': self.grade,
            'project': self.project,
            'description': self.description,
            'channel': self.channel,
            'student_type': self.student_type,
            'intention': self.intention
        }

        return ' '.join([f"{key}: {value}" for key, value in data_dict.items() if
                                                  key not in ['id', 'intention']])

    def data_entry(self):
        return self.id, self.text, self.label


if __name__ == '__main__':
    dataset = BusinessIntentionDataset('.')
    print(dataset[0])
