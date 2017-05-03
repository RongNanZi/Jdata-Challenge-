# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os


from datetime import datetime
import numpy as np
import operator 
import tensorflow as tf
from CnnLSTMmodel import CnnLSTM
import component

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_palcement=True))

batch_size = 256

data_path = './data/'
logs_path = './logs'
model_saver_path = './existed_model/CNNLSTM/model.ckpt'
DAYS = (datetime(2016,4,15)-datetime(2016,2,1)).days + 1
MAX_times = 8 

x = np.load('./data/x.data')
x_times = np.load('./data/x_times.data')
y = np.load('./data/y.data')


sku2no = dict( np.load('./data/sku.voc').tolist() )
no2sku = dict( zip(sku2no.values(), sku2no.keys()))

brand_voc = np.load('./data/brand.voc').tolist()


       
model = CnnLSTM(days = DAYS,
                      max_time = MAX_times,
                      nums_feature = 14,
                      skuid_voc_size = len(sku2no),
                      brand_voc_size = len(brand_voc)+1)

train_op = component.training_graph(model.losses,
                                    learning_rate=0.1)

tf.summary.scalar("loss", model.losses)
tf.summart.scalar('accurate',model.acc)
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()

def train():
    tf.initialize_all_variables().run()  

    num_batches_per_epoch = int(len(x) / batch_size)
    summary_train_writer = tf.summary.FileWriter(logs_path, graph = sess.graph)
    #summary_train_writer = tf.summary.FileWriter(logs_path)
    #saver.restore(sess, model_saver_path)
    for epoch in range(50):
        start = datetime.now()
        for j in range(num_batches_per_epoch):
            idx = j * batch_size
            
            _, summary = sess.run([train_op,  merged_summary_op ], 
                                            feed_dict={model.features: x[idx:(idx+batch_size)],
                                                       model.event_size: x_times[idx:(idx+batch_size)],
                                                       model.input_y: y[idx:(idx+batch_size)] })
            summary_train_writer.add_summary(summary,epoch*num_batches_per_epoch+j)
        end = datetime.now()
        print('process one epoch cost time is:' + str(end - start))

    summary_train_writer.close()
   
    if not os.path.exists(model_saver_path):
        os.makedirs(model_saver_path)
    saver_path = saver.save(sess, model_saver_path)

def test():
    goals = []
    num_batches_per_epoch = int(len(x) / batch_size)
    for j in range(num_batches_per_epoch):
        idx = j * batch_size
        predict = sess.run([model.predict], 
                           feed_dict={model.features: x[idx:(idx+batch_size)],
                                      model.event_size: x_times[idx:(idx+batch_size)],
                                      model.input_y: y[idx:(idx+batch_size)] })
    
        predict = predict[0]
        goal = [no2sku[item] for item in predict.tolist()]        
        goals = operator.concat(goals, goal)
        
    idx = num_batches_per_epoch * batch_size
    predict = sess.run([model.predict], 
                       feed_dict={model.features: x[idx:],
                                  model.event_size: x_times[idx:],
                                  model.input_y: y[idx:] })
    predict = predict[0]
    goal = [no2sku[item] for item in predict.tolist()]        
    goals = operator.concat(goals, goal)
    
    goals = np.asarray(goals)
    goals.dump(data_path + 'goal.list')
    
    
train()
test()






























