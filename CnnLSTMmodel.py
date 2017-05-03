# -*- coding: utf-8 -*-
import tensorflow as tf
from model.feild_embedding import Embedding
import component 
class CnnLSTM(object):
    
    def __init__(self,
                 days,
                 max_time,
                 nums_feature,
                 skuid_voc_size,
                 brand_voc_size,
                 rnn_size = 256,
                 rnn_layers = 1,
                 isTrain = True,
                 ):

        self.rnn_size = rnn_size
        

        
        with tf.name_scope('Input-layer'):
            # the input's shape is (batchsize, days from Feb to Apl, events in everyday, numbers of features in one event),every feature is a number
            self.features = tf.placeholder(shape = [None, days, max_time, nums_feature], dtype = tf.int64, name = 'features')
            self.event_size = tf.placeholder(shape = [None, days], dtype = tf.int64, name = 'numbers_events_day')
            self.input_y = tf.placeholder(shape = [None, days], dtype = tf.int64, name = 'target')
        
        #shape is (batchsize, days, max_time, embeddingsum)
        cnn_input = Embedding(self.features, nums_feature,skuid_voc_size,brand_voc_size).embedding_vec
        
        '''
        encoding the events in everyday at one user
        '''
        with tf.variable_scope('cnnEncode'):
            #shape is (batchsize, days, 1, outdims)
            encode_day = component.conv2d(cnn_input,
                             output_dim=64,
                             k_h=1,
                             k_w=max_time)
            encode_day = tf.squeeze(encode_day,
                                    axis=[2])
        
            
        with tf.name_scope('daysLSTM'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size, state_is_tuple= False)
            if isTrain:
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=0.5)
            if rnn_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([cell] * rnn_layers, state_is_tuple=False)
                
            states = tf.unstack(encode_day, axis = 1)
            outputs, _ = tf.contrib.rnn.static_rnn(cell,
                                                  states,
                                                  dtype=tf.float32)
        with tf.variable_scope('logit'):
            targets = tf.unstack(self.input_y, axis = 1)
            losses = []
            accs = []
            for output,target in zip(outputs[:-5],targets):
                logit = tf.contrib.layers.fully_connected(output, skuid_voc_size, activation_fn=None)#logit's shape is (batchsize, skuid_voc_size)
                acc = component.accuracy_graph(logit, target)
                loss = component.loss_graph(logit, target)
                losses.append(loss)
                accs.append(acc)
            self.losses = sum(losses)
            self.acc = sum(accs)/len(accs)
        
        '''
        generate the final goal,the 'logit' variable must be reused
        '''
        with tf.variable_scope('logit',reuse = True):
            pre_logit = tf.contrib.layers.fully_connected(outputs[-1], skuid_voc_size, activation_fn=None)
            self.predict = tf.argmax(pre_logit, -1)
            
        
         
                 

