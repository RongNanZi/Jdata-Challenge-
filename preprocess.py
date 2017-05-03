# -*- coding: utf-8 -*-
import pandas as pd

raw_file_names = ['JData_Action_201602.csv', 'JData_Action_201603.csv', 'JData_Action_201604.csv']
merg_name = 'JData_Action.csv'

csv_shapes = []
for file_name in raw_file_names:
    cvs_data = pd.read_csv('./data/raw/' + file_name)
    csv_shapes.append(cvs_data.shape)
    cvs_data = cvs_data.drop_duplicates(subset =['user_id', 'sku_id', 'time'])
    csv_shapes.append(cvs_data.shape)
    cvs_data.to_csv('./data/' + file_name)

merger_csv_data = pd.read_csv('./data/' + raw_file_names[0])
for file_name in raw_file_names[1:]:
    csv_data = pd.read_csv('./data/' + file_name)
    merger_csv_data = pd.concat([merger_csv_data, csv_data],0, ignore_index = False)
merger_csv_data.to_csv('./data/' + merg_name)
