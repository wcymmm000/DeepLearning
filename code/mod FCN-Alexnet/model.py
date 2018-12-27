# -*- coding:UTF-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def unet(x, is_train=True, reuse=False):
    batch_size=FLAGS.batch_size
    n_out=1
    pad='SAME'
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    input_size=256
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')

        conv = Conv2d(inputs, 96, (11,11), (4,4), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv = LocalResponseNormLayer(conv,4,bias=1,alpha=0.001/9.0,beta=0.75,name='lrn1')
        conv = MaxPool2d(conv, (2,2), padding='SAME', name='pool1')
        conv = Conv2d(conv, 256, (5,5), (1,1), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv = LocalResponseNormLayer(conv,4,bias=1,alpha=0.001/9.0,beta=0.75,name='lrn2')
        conv = MaxPool2d(conv, (2,2), padding='SAME', name='pool2')
        conv = Conv2d(conv, 384, (3,3), (1,1), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv = Conv2d(conv, 384, (3,3), (1,1), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv = Conv2d(conv, 256, (3,3), (1,1), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv = Conv2d(conv, 128, (1,1), (1,1), act=tf.nn.relu, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool3')
        out = DeConv2d(conv, 1, (1,1), out_size=(FLAGS.output_size, FLAGS.output_size), strides=(32,32),
                 padding=pad, act=tf.nn.sigmoid, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv')
      
                 
     
        
    return  out.outputs

