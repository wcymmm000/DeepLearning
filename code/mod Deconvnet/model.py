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

        conv = Conv2d(inputs, 64, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv11')
        conv = Conv2d(conv, 64, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv12')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool1')
        conv = Conv2d(conv, 128, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv21')
        conv = Conv2d(conv, 128, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv22')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool2')
        conv = Conv2d(conv, 256, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv31')
        conv = Conv2d(conv, 256, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv32')
        conv = Conv2d(conv, 256, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv33')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool3')
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv41')
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv42')
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv43')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool4')        
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv51')
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv52')
        conv = Conv2d(conv, 512, (3,3), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv53')
        conv = MaxPool2d(conv, (2, 2), padding='SAME', name='pool5')   
        #conv = Conv2d(conv, 4096, (7,7), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv61')
        conv = Conv2d(conv, 1024, (1,1), (1,1), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv62')
        
        deconv = DeConv2d(conv, 512, (7,7), out_size=(FLAGS.output_size/32, FLAGS.output_size/32), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        deconv = UpSampling2dLayer(deconv, (2,2), is_scale=True, method=0, align_corners=False, name='upsample2d_layer5')
        deconv = DeConv2d(deconv, 512, (3,3), out_size=(FLAGS.output_size/16, FLAGS.output_size/16), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv51')
        deconv = DeConv2d(deconv, 512, (3,3), out_size=(FLAGS.output_size/16, FLAGS.output_size/16), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv52')
        deconv = DeConv2d(deconv, 512, (3,3), out_size=(FLAGS.output_size/16, FLAGS.output_size/16), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv53')
        deconv = UpSampling2dLayer(deconv,(2,2), is_scale=True, method=0, align_corners=False, name='upsample2d_layer4')
        deconv = DeConv2d(deconv, 512, (3,3), out_size=(FLAGS.output_size/8, FLAGS.output_size/8), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv41')
        deconv = DeConv2d(deconv, 512, (3,3), out_size=(FLAGS.output_size/8, FLAGS.output_size/8), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv42')
        deconv = DeConv2d(deconv, 256, (3,3), out_size=(FLAGS.output_size/8, FLAGS.output_size/8), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv43')
        deconv = UpSampling2dLayer(deconv, (2,2), is_scale=True, method=0, align_corners=False, name='upsample2d_layer3')
        deconv = DeConv2d(deconv, 256, (3,3), out_size=(FLAGS.output_size/4, FLAGS.output_size/4), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv31')
        deconv = DeConv2d(deconv, 256, (3,3), out_size=(FLAGS.output_size/4, FLAGS.output_size/4), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv32')
        deconv = DeConv2d(deconv, 128, (3,3), out_size=(FLAGS.output_size/4, FLAGS.output_size/4), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv33')       
        deconv = UpSampling2dLayer(deconv, (2,2), is_scale=True, method=0, align_corners=False, name='upsample2d_layer2')
        deconv = DeConv2d(deconv, 128, (3,3), out_size=(FLAGS.output_size/2, FLAGS.output_size/2), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv21')
        deconv = DeConv2d(deconv, 64, (3,3), out_size=(FLAGS.output_size/2, FLAGS.output_size/2), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv22')
        deconv = UpSampling2dLayer(deconv, (2,2), is_scale=True, method=0, align_corners=False, name='upsample2d_layer1')
        deconv = DeConv2d(deconv, 64, (3,3), out_size=(FLAGS.output_size, FLAGS.output_size), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv11')
        deconv = DeConv2d(deconv, 32, (3,3), out_size=(FLAGS.output_size, FLAGS.output_size), strides=(1,1),
                 padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv12')         
                 
        out = Conv2d(deconv, n_out, (1, 1), act=tf.nn.sigmoid, name='out')      
        
    return  out.outputs

