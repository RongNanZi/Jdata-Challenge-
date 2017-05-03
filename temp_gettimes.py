# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np

data_path = './data/'
DAYS = (datetime(2016,2,29)-datetime(2016,2,1)).days + 1
MAX_times = 16 
     
main_table = pd.read_csv(data_path + 'main_table.csv', encoding = 'gbk')
main_table[['time']] = main_table[['time']].apply(pd.to_datetime)

COLUMNS = ['age','sex','user_lv_cd',
           'sku_id','model_id','type','cate','brand',
           'comment_num','has_bad_comment','bad_comment_rate',
           'a1','a2','a3']
NULL_VALUES = [[0,0,0,
               0,0,0,0,0,
               0,0,0,
               0,0,0]]
data_x_times = []
def get_everyuser(df):
    #start = datetime.now()
    
   
    data_x_times_temp = []
   
    for today in range(DAYS):
        this_day = (datetime(2016,2,1) + timedelta(days = today))
        find_user_day = df[df.time == this_day]
        if find_user_day.empty :
            wanted_length = 1
        elif find_user_day.shape[0] < MAX_times:
            wanted_length = find_user_day.shape[0]
        else:
            wanted_length = MAX_times
            
        data_x_times_temp.append(wanted_length)

        
    
          
    data_x_times.append(data_x_times_temp)
    
    #end = datetime.now()
    #print('process one user cost time is:' + str(end - start))
main_table.groupby('user_id').apply(get_everyuser)
data_x_times = np.asarray(data_x_times)
data_x_times.dump(data_path + 'x_times.data')