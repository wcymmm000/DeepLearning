# -*- coding:UTF-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np

from tensorlayer.layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

def unet(x1,x2, is_train=True, reuse=False):
    batch_size=FLAGS.batch_size
    n_out=1
    pad='SAME'
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("u_net", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        inputs_1 = InputLayer(x1, name='inputs_1')
        inputs_2 = InputLayer(x2, name='inputs_2')
        inputs = ConcatLayer([inputs_1, inputs_2], concat_dim=3, name='inputs')
        conv1 = Conv2d(inputs, 64, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv1')
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn1')
        
        conv2 = Conv2d(conv1, 128, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv2')
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(conv2, 256, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv3')
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn3')

        conv4 = Conv2d(conv3, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv4')
        conv4 = BatchNormLayer(conv4, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn4')

        conv5 = Conv2d(conv4, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv5')
        conv5 = BatchNormLayer(conv5, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn5')

        conv6 = Conv2d(conv5, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv6')
        conv6 = BatchNormLayer(conv6, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn6')

        conv7 = Conv2d(conv6, 512, (4, 4), (2, 2), act=None, padding=pad, W_init=w_init, b_init=b_init, name='conv7')
        conv7 = BatchNormLayer(conv7, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train, gamma_init=gamma_init, name='bn7')

        conv8 = Conv2d(conv7, 1024, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2), padding=pad, W_init=w_init, b_init=b_init, name='conv8')
        print(" * After conv: %s" % conv8.outputs)
        # exit()
        # print(nx/8)
        up7 = DeConv2d(conv8, 1024, (4, 4), out_size=(FLAGS.output_size/128, FLAGS.output_size/128), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv7')
        up7 = BatchNormLayer(up7, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn7')

        # print(up6.outputs)
        up6 = ConcatLayer([up7, conv7], concat_dim=3, name='concat6')
        up6 = DeConv2d(up6, 1024, (4, 4), out_size=(FLAGS.output_size/64, FLAGS.output_size/64), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv6')
        up6 = BatchNormLayer(up6, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn6')
        # print(up6.outputs)
        # exit()

        up5 = ConcatLayer([up6, conv6], concat_dim=3, name='concat5')
        up5 = DeConv2d(up5, 1024, (4, 4), out_size=(FLAGS.output_size/32, FLAGS.output_size/32), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv5')
        up5 = BatchNormLayer(up5, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn5')
        # print(up5.outputs)
        # exit()

        up4 = ConcatLayer([up5, conv5] ,concat_dim=3, name='concat4')
        up4 = DeConv2d(up4, 1024, (4, 4), out_size=(FLAGS.output_size/16, FLAGS.output_size/16), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv4')
        up4 = BatchNormLayer(up4, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn4')

        up3 = ConcatLayer([up4, conv4] ,concat_dim=3, name='concat3')
        up3 = DeConv2d(up3, 256, (4, 4), out_size=(FLAGS.output_size/8, FLAGS.output_size/8), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn3')

        up2 = ConcatLayer([up3, conv3] ,concat_dim=3, name='concat2')
        up2 = DeConv2d(up2, 128, (4, 4), out_size=(FLAGS.output_size/4, FLAGS.output_size/4), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn2')

        up1 = ConcatLayer([up2, conv2] ,concat_dim=3, name='concat1')
        up1 = DeConv2d(up1, 64, (4, 4), out_size=(FLAGS.output_size/2, FLAGS.output_size/2), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn1')

        up0 = ConcatLayer([up1, conv1] ,concat_dim=3, name='concat0')
        up0 = DeConv2d(up0, 64, (4, 4), out_size=(FLAGS.output_size, FLAGS.output_size), strides=(2, 2),
                                    padding=pad, act=None, batch_size=batch_size, W_init=w_init, b_init=b_init, name='deconv0')
        up0 = BatchNormLayer(up0, act=tf.nn.relu, is_train=is_train, gamma_init=gamma_init, name='dbn0')
        # print(up0.outputs)
        # exit()

        out = Conv2d(up0, n_out, (1, 1), act=tf.nn.sigmoid, name='out')      
        
    return  out.outputs

