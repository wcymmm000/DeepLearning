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
    nx=FLAGS.output_size
    ny=FLAGS.output_size
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    def denseblock(net,kernel_num=64,name='DenseBlock'):
	  with tf.variable_scope(name):
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn1')
        net = Conv2d(net, kernel_num, (1,1), act=tf.nn.relu, name='cv1')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')
        net = Conv2d(net, kernel_num, (3,3), act=tf.nn.relu, name='cv2')
	  return net
    def densehub(net,kernel_num=64,depth=5,name='densehub'):
		with tf.variable_scope(name, reuse=reuse):
        for layernum in range(0,depth):
			if layernum==0:
			    concat=net
			else:
			    concat=ConcatLayer([concat,net] , 3, name='c%d' % layernum)
			net = denseblock(net,kernel_num,name='d%d' % layernum)
	    return net
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs = InputLayer(x, name='inputs')
        conv1 = densehub(inputs,32,3,name='conv1')
        pool1 = MaxPool2d(conv1, (2, 2), name='pool1')
        conv2 = densehub(pool1,64,3,name='conv2')
        pool2 = MaxPool2d(conv2, (2, 2), name='pool2')
        conv3 = densehub(pool2,128,3,name='conv3')
        pool3 = MaxPool2d(conv3, (2, 2), name='pool3')
        conv4 = densehub(pool3,256,3,name='conv4')
        pool4 = MaxPool2d(conv4, (2, 2), name='pool4')
        conv5 = densehub(pool4,256,3,name='conv5')
        up4 = DeConv2d(conv5, 256, (3, 3), (nx/8, ny/8), (2, 2), name='deconv4')
        up4 = ConcatLayer([up4, conv4], 3, name='concat4')
        conv4 = densehub(up4,256,3,name='uconv4')
        up3 = DeConv2d(conv4, 128, (3, 3), (nx/4, ny/4), (2, 2), name='deconv3')
        up3 = ConcatLayer([up3, conv3], 3, name='concat3')
        conv3 = densehub(up3,128,3,name='uconv3')
        up2 = DeConv2d(conv3, 64, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        up2 = ConcatLayer([up2, conv2], 3, name='concat2')
        conv2 = densehub(up2,64,3,name='uconv2')
        up1 = DeConv2d(conv2, 32, (3, 3), (nx/1, ny/1), (2, 2), name='deconv1')
        up1 = ConcatLayer([up1, conv1] , 3, name='concat1')
        conv1 = densehub(up1,32,3,name='uconv1')

        out = Conv2d(conv1, n_out, (1, 1), act=tf.nn.sigmoid, name='out')      
        
    return  out.outputs
