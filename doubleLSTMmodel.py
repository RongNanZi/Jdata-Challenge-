# -*- coding: utf-8 -*-
import tensorflow as tf
from model.feild_embedding import Embedding
'''
COLUMNS = ['age','sex','user_lv_cd',
           'sku_id','model_id','type','cate','brand',
           'comment_num','has_bad_comment','bad_comment_rate',
           'a1','a2','a3']
'''
           
class DoubleLSTM(object):
    
    def __init__(self,
                 days,
                 max_time,
                 nums_feature,
                 skuid_voc_size,
                 brand_voc_size,
                 rnn_size = 256,
                 rnn_layers = 2,
                 drop_pro = 0.5,
                 ):

        self.rnn_size = rnn_size
        self.drop_pro = drop_pro

        
        with tf.name_scope('Input-layer'):
            # the input's shape is (batchsize, days from Feb to Apl, events in everyday, numbers of features in one event),every feature is a number
            self.features = tf.placeholder(shape = [None, days, max_time, nums_feature], dtype = tf.int64, name = 'features')
            self.event_size = tf.placeholder(shape = [None, days], dtype = tf.int64, name = 'numbers_events_day')
            self.input_y = tf.placeholder(shape = [None, days], dtype = tf.int64, name = 'target')
        
        rnn_input = Embedding(self.features, nums_feature,skuid_voc_size,brand_voc_size).embedding_vec
        
        input_shape = rnn_input.get_shape().as_list()
        rnn_input = tf.reshape(rnn_input, [-1, max_time, input_shape[-1]])#shape is (batchsize*days, max_time, embeddingsum)
        #rnn_input = tf.unstack(rnn_input, axis = 2) # none number shape is (batchsize, days, embeddigsum)
        #event_shape = self.event_size.get_shape().as_list()
        rnn_event_size = tf.reshape(self.event_size, [-1])
        with tf.name_scope('encodingLSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, state_is_tuple= True)
            _, states = tf.nn.dynamic_rnn(cell,
                                       rnn_input,
                                       sequence_length = rnn_event_size,
                                       dtype=tf.float32)
            #the states' shape is (batchsize*days, rnn_size)
            states = tf.reshape(states[1],[-1, input_shape[1], self.rnn_size])
            
        with tf.name_scope('daysLSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, state_is_tuple= False)
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.5)
            if rnn_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layers, state_is_tuple=False)
                
            states = tf.unstack(states, axis = 1)
            outputs, _ = tf.contrib.rnn.static_rnn(cell,
                                                  states,
                                                  dtype=tf.float32)
        with tf.name_scope('logits_probs'):
            targets = tf.unstack(self.input_y, axis = 1)
            logits = []
            self.losses = []
            #outputs' shape is days*(batchsize, rnnsize)
            with tf.variable_scope('logit'):
                for output, target in zip(outputs[:-5], targets[:-5]):
                    logit = tf.contrib.layers.fully_connected(output, skuid_voc_size, activation_fn=None)#logit's shape is (batchsize, skuid_voc_size)
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=target), name='loss')
                    logits.append(logit)
                    self.losses.append(loss)
            logits = tf.stack(logits, axis = 1)
            self.losses = tf.stack(self.losses)
            
            '''
            generate the final goal,the 'logit' variable must be reused
            '''
            with tf.variable_scope('logit',reuse = True):
                pre_logit = tf.contrib.layers.fully_connected(outputs[-1], skuid_voc_size, activation_fn=None)
                self.predict = tf.argmax(pre_logit, -1)
            
        with tf.name_scope('loss'):
            self.losses = tf.reduce_mean(self.losses)
         
             
         
                 













