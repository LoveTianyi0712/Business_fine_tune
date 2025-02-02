# -*- coding: utf-8 -*-
# @Time    : 2025/1/24 16:31
# @Author  : Gan Liyifan
# @File    : dataprocesser.py
import pandas as pd
import json
import warnings

warnings.filterwarnings('ignore')


def load_data(file_name='Q2表彰数据明细.xlsx', sheet_name='商机明细'):
    data = pd.read_excel(file_name, sheet_name)
    return data


def classify_quality(intention_level):
    if intention_level in ['高', '中']:
        return '高质量商机'
    elif intention_level in ['低', '无']:
        return '低质量商机'
    else:
        return '未知商机'


def clean_data(data):
    history_conditions = ['已成单', '成单归档', '死单归档', '无效归档']
    history_data = data[data['商机状态'].isin(history_conditions)]
    future_data = data[~data['商机状态'].isin(history_conditions)]
    history_data['商机状态'] = history_data['商机状态'].replace('已成单', '成单归档')
    history_data['商机质量'] = history_data['意向等级'].apply(classify_quality)
    history_data.loc[history_data['商机状态'] == '成单归档', '商机质量'] = '高质量商机'

    history_data = history_data.dropna(subset=['意向等级'])
    history_data['在读年级'] = history_data['在读年级'].fillna('未知')
    history_data['意向项目'] = history_data['意向项目'].fillna('未知')
    history_data['业绩一级渠道'] = history_data['业绩一级渠道'].fillna('未知')
    history_data['历史跟进内容'] = history_data['历史跟进内容'].fillna('无')
    history_data['新老生'] = history_data['新老生'].fillna('未知')

    future_data['在读年级'] = future_data['在读年级'].fillna('未知')
    future_data['意向项目'] = future_data['意向项目'].fillna('未知')
    future_data['业绩一级渠道'] = future_data['业绩一级渠道'].fillna('未知')
    future_data['历史跟进内容'] = future_data['历史跟进内容'].fillna('无')
    future_data['新老生'] = future_data['新老生'].fillna('未知')

    return history_data, future_data


def dump_data_json(data, output_file='output.json'):
    columns = ['商机编码', '在读年级', '意向项目', '历史跟进内容', '业绩一级渠道', '新老生', '意向等级']
    extracted_data = data[columns]

    data_list = extracted_data.to_dict(orient='records')

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            json_item = {
                "id": item['商机编码'],
                "grade": item['在读年级'],
                "project": item['意向项目'],
                "description": item['历史跟进内容'],
                "channel": item['业绩一级渠道'],
                "student_type": item['新老生'],
                "intention": item['意向等级'],
            }
            f.write(json.dumps(json_item, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data = load_data()
    history_data, future_data = clean_data(data)
    dump_data_json(history_data, 'train/history_data.json')
    dump_data_json(future_data, 'test/future_data.json')