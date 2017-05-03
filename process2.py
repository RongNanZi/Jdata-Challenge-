# -*- coding: utf-8 -*-
import pandas as pd
from datetime import datetime
from datetime import timedelta
import collections
import numpy as np

data_path = './data/'
DAYS = (datetime(2016,4,15)-datetime(2016,2,1)).days + 1
MAX_times = 8 
     
main_table = pd.read_csv(data_path + 'main_table.csv', encoding = 'gbk')
main_table[['time']] = main_table[['time']].apply(pd.to_datetime)

# remove the unnessary column

main_table = main_table.drop('Unnamed: 0',axis = 1)

#change the age projection
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
date_age = main_table[['age']].values
for i in range(len(date_age)):
     date_age[i][0]= age2int(date_age[i][0])
main_table[['age']] = pd.DataFrame(data = date_age, columns = ['age'])



sku_counter = collections.Counter(main_table[['sku_id']].values.ravel())
sku_voc_inv = [x[0] for x in sku_counter.most_common()]
sku_voc_inv = list(sorted(sku_voc_inv))
sku_voc = {x: (i+1) for i, x in enumerate(sku_voc_inv)}
sku_voc.update({0 : 0})

brand_counter = collections.Counter(main_table[['brand']].values.ravel())
brand_voc_inv = [x[0] for x in brand_counter.most_common()]
brand_voc_inv = list(sorted(brand_voc_inv))
brand_voc = {x: (i+1) for i, x in enumerate(brand_voc_inv)}

#find the new value in sku vocabulary 
data_sku = main_table[['sku_id']].values
for i in range(len(data_sku)):
    data_sku[i][0] = sku_voc[data_sku[i][0]]
main_table[['sku_id']] = pd.DataFrame(data = data_sku, columns = ['sku_id'])


#find the new value in brand vocabulary
data_brand = main_table[['brand']].values
for i in range(len(data_brand)):
    data_brand[i][0] = brand_voc[data_brand[i][0]]
main_table[['brand']] = pd.DataFrame(data = data_brand, columns = ['brand'])

#main_table.to_csv(data_path + 'main_table_afterfilter.csv')

COLUMNS = ['age','sex','user_lv_cd',
           'sku_id','model_id','type','cate','brand',
           'comment_num','has_bad_comment','bad_comment_rate',
           'a1','a2','a3']
main_table[['sex']] = main_table[['sex']].applymap(lambda x:(2-x))

NULL_VALUES = [[0,0,0,
               0,0,0,0,0,
               0,0,0,
               0,0,0]]

data_x = []
data_x_times = []
data_y = []


empty_frame = pd.DataFrame(data = NULL_VALUES,columns = COLUMNS,dtype = int)

def get_everyuser(df):
    #start = datetime.now()
    
    data_x_temp = []
    data_x_times_temp = []
    data_y_temp = []


    for today in range(DAYS):
        this_day = (datetime(2016,2,1) + timedelta(days = today))
        find_user_day = df[df.time == this_day]
        if find_user_day.empty :
            wanted_data = empty_frame
        else:
            wanted_data = find_user_day[COLUMNS]
            
        wanted = np.zeros([MAX_times,14],dtype = int)
        wanted_length = wanted_data.shape[0]
        if wanted_length > MAX_times:
            wanted = wanted_data.values[:MAX_times]
            wanted_length = MAX_times
        else:
            wanted[:wanted_length] = wanted_data.values
                  
        data_x_temp.append(wanted) 
        data_x_times_temp.append(wanted_length)

        #find the target
        shop = df[(df.time > this_day) & (df.time  <= (this_day + timedelta(days = 5)))]
        shop = shop[shop.type == 6]
        if shop.empty:
            data_y_temp.append(0)
        else:
            data_y_temp.append(shop.values[0][1])
    
    data_x.append(data_x_temp)        
    data_x_times.append(data_x_times_temp)
    data_y.append(data_y_temp)
    #end = datetime.now()
    #print('process one user cost time is:' + str(end - start))
main_table.groupby('user_id').apply(get_everyuser)

user_list = [key for key,_ in main_table.groupby('user_id')]


data_x = np.asarray(data_x)
data_x_times = np.asarray(data_x_times)
data_y = np.asarray(data_y)
sku_voc = np.asarray(sku_voc)
brand_voc = np.asarray(brand_voc)
user_list = np.asarray(user_list)


data_x.dump(data_path + 'x.data')
data_x_times.dump(data_path + 'x_times.data')
data_y.dump(data_path + 'y.data')
sku_voc.dump(data_path + 'sku.voc')
brand_voc.dump(data_path + 'brand.voc')
user_list.dump(data_path + 'user.list')



