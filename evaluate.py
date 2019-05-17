# -*- coding:UTF-8 -*-

import os, sys, pprint, time
import scipy.misc as misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
import skimage
import random
import natsort

from model import *
import scipy.io


flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("batch_size", 1, "The number of batch images ")
flags.DEFINE_integer("image_size", 512, "The size of image to use (will be center cropped) ")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")
flags.DEFINE_string("feature_dir", "features", "Directory name to save the image samples ")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing ")

FLAGS = flags.FLAGS

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],[resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)


def main(_):

        

    with tf.variable_scope("gen"):
      ## 占位符声明
      input_img = tf.placeholder(tf.float32, [None,None,None,1], name='input_img')
      g_logits = generator(input_img)
        
        
    ## 开启会话
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess) 
    saver.restore(sess,'./checkpoint/model.ckpt')
    rawpic=[]
    sample_file='./10.bmp'
    oimg=get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0)
    oimg=oimg[:,:,0]
    oimg = (oimg+1)/2
    img=oimg[np.newaxis,:,:,np.newaxis]
             
    outimg = sess.run([g_logits], feed_dict={input_img : img})
    outimg = np.array(outimg).astype(np.float32)
    outimg=np.squeeze(outimg)
    outimg=outimg[np.newaxis,:,:,np.newaxis]
    print(img.shape)
    print(outimg.shape)
    
    img = np.concatenate((img,outimg,np.zeros((FLAGS.batch_size,FLAGS.output_size,FLAGS.output_size,1))), axis = 3)
    tl.visualize.save_images(img, [1,1], './seg_10.bmp')

    
              
       
        
if __name__ == '__main__':
    tf.app.run()
