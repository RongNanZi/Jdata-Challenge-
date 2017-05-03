# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime

data_path = './data/'

product = pd.read_csv(data_path + '/raw/JData_Product.csv')
comment = pd.read_csv(data_path + '/raw/JData_Comment.csv')
user = pd.read_csv(data_path + '/raw/JData_User.csv', encoding = 'gbk')
action = pd.read_csv(data_path + '/JData_Action.csv')

def a2int(a):
    if a == -1:
        return 0
    else:
        return a
for item in ['a1', 'a2', 'a3']:
    data_a = product[[item]].values
    for i in range(len(data_a)):
        data_a[i][0] = a2int(data_a[i][0])
    product[[item]] =  pd.DataFrame(data = data_a, columns = [item])


        

comment[['bad_comment_rate']] = comment[['bad_comment_rate']] * 100
comment[['bad_comment_rate']] = comment[['bad_comment_rate']].astype(int)
comment[['dt']] = comment[['dt']].apply(pd.to_datetime)

#user[['sex']] = user[['sex']].astype(int)
'''
def age2int(age):
    if '26-35' in age:
        return 1
    elif '46-55' in age:
        return 2
    elif '36-45' in age:
        return 3
    elif '16-25' in age:
        return 4
    elif '56' in age:
        return 5
    elif '15' in age:
        return 6
    else:
        return 0
data_age = user[['age']].values
for i in range(len(data_age)):
     data_age[i][0]= age2int(data_age[i][0])
user[['age']] = pd.DataFrame(data = data_age, columns = ['age'])
'''


#action[['user_id','model_id']] = action[['user_id','model_id']].astype(int)
action[['user_id']] = action[['user_id']].astype(int)
action[['time']] = action[['time']].apply(pd.to_datetime)
date_time = action[['time']].values
for i in range(len(date_time)):
     date_time[i][0]= pd.Timestamp(date_time[i][0]).date()
action[['time']] = pd.DataFrame(data = date_time, columns = ['time'])
action[['model_id']] = action[['model_id']] + 1
action = action[action.time > datetime(2016,1,31)]





COLUMNS = ['age','sex','user_lv_cd',
           'sku_id','model_id','type','cate','brand',
           'comment_num','has_bad_comment','bad_comment_rate'
           'a1','a2','a3']
main_table = pd.merge(action[['user_id','sku_id','time','model_id','type','cate','brand']], 
                      user[['user_id','age','sex','user_lv_cd']], 
                      on = 'user_id',
                      how = 'left')
main_table = pd.merge(main_table, 
                      product[['sku_id', 'a1', 'a2', 'a3']], 
                      on = 'sku_id', 
                      how = 'left')
comment[['time']] = comment[['dt']]
main_table = pd.merge(main_table, 
                      comment[['sku_id', 'comment_num', 'has_bad_comment', 'bad_comment_rate','time']], 
                      on = ['sku_id','time'], 
                      how = 'left')

main_table = main_table.fillna(value={'model_id':0, 'sex':2, 'age':0, 'a1':0,'a2':0,'a3':0, 'comment_num':0,'has_bad_comment':0,'bad_comment_rate':0})



need_int_columns = ['sku_id','model_id','type','cate','brand','sex','a1','a2','a3','comment_num','has_bad_comment','bad_comment_rate']
main_table[need_int_columns] = main_table[need_int_columns].astype(int)
main_table.to_csv(data_path + 'main_table.csv', encoding = 'gbk')
