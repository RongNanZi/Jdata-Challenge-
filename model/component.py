# -*- coding: utf-8 -*-
import tensorflow as tf
from math import sqrt

def conv2d(input_, output_dim, k_h, k_w, name="conv2d"):
    with tf.variable_scope(name):
        #the input_.get_shape()[-1] is input channel or input_dim
        filter_shape = [k_h, k_w, input_.get_shape().as_list()[-1], output_dim]

        stdv = 1/sqrt(output_dim * k_h)
        W = tf.Variable(tf.random_uniform(shape=filter_shape,minval=-stdv, maxval=stdv), name='W') 
        # The kernel of the conv layer is a trainable vraiable
        b = tf.Variable(tf.random_uniform(shape=[output_dim], minval=-stdv, maxval=stdv), name = 'b')
        # and the biases as well
        conv = tf.nn.conv2d(input_, W, strides=[1, 1, input_.get_shape().as_list()[2], 1], padding='VALID')
    return  tf.nn.bias_add(conv, b)

'''
built loss graph
the targets's shape is {batch_size}
'''
def loss_graph(logits, targets):

    with tf.variable_scope('Loss'):

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets), name='loss')
    return loss

def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('Adam_Training'):
        # ADAM learning parameter
        learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return train_op

def model_size():

    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape().as_list():
            sz *= dim.value
        size += sz
    return size

def accuracy_graph(logist, targets):
    
    with tf.name_scope('Accuracy'):
     
        acc = tf.equal(tf.argmax(logist, 1), targets)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc

