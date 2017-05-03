# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from datetime import datetime
import numpy as np
import operator 
import tensorflow as tf
from dynamicLSTM import dynamicLSTM
sess = tf.InteractiveSession()

batch_size = 64
data_path = './data/'
logs_path = './logs'
model_saver_path = './existed_model/model.ckpt'
DAYS = (datetime(2016,4,15)-datetime(2016,2,1)).days + 1
MAX_times = 8 




x = np.load('./data/x.data')
x_times = np.load('./data/x_times.data')
y = np.load('./data/y.data')

sku2no = dict( np.load('./data/sku.voc').tolist() )
no2sku = dict( zip(sku2no.values(), sku2no.keys()))
brand_voc = np.load('./data/brand.voc').tolist()


       
d_model = dynamicLSTM(days = DAYS,
                      max_time = MAX_times,
                      nums_feature = 14,
                      skuid_voc_size = len(sku2no),
                      brand_voc_size = len(brand_voc)+1)




saver = tf.train.Saver()


def test():
    saver.restore(sess, model_saver_path)
    goals = []
    num_batches_per_epoch = int(len(x) / batch_size)
    for j in range(num_batches_per_epoch):
        if j == 2:
            print('test is ok!')
        idx = j * batch_size
        predict = sess.run([d_model.predict], 
                           feed_dict={d_model.features: x[idx:(idx+batch_size)],
                                      d_model.event_size: x_times[idx:(idx+batch_size)],
                                      d_model.input_y: y[idx:(idx+batch_size)] })
    
        predict = predict[0]
        goal = [no2sku[item] for item in predict.tolist()]        
        goals = operator.concat(goals, goal)
        
    idx = num_batches_per_epoch * batch_size
    predict = sess.run([d_model.predict], 
                       feed_dict={d_model.features: x[idx:],
                                  d_model.event_size: x_times[idx:],
                                  d_model.input_y: y[idx:] })
    goal = [no2sku[item] for item in predict.tolist()]        
    goals = operator.concat(goals, goal)
    
    goals = np.asarray(goals)
    goals.dump(data_path + 'goal.list')
test()

