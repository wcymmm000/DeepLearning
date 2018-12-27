import os, sys, pprint, time
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model import *
from utils import *
import scipy.io

#pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate face image.

Usage : see README.md
"""
flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [21]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0001]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 4, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 512, "The size of image to use (will be center cropped) [100]")
flags.DEFINE_integer("output_size", 512, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_step", 2371, "The interval of generating sample. [500]")
flags.DEFINE_string("sample_dir", "samples2", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
    
    resue_train=0;
    
    #pp.pprint(flags.FLAGS.__flags)
	
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    with tf.device("/gpu:0"):
        ##========================= DEFINE MODEL ===========================##
        z = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,1], name='z_noise')
        real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,1], name='real_images')
		
        # z --> generator for training
        g_logits = generator_simplified_api(z, is_train=True, reuse=False)
        print(g_logits.shape)
        g_loss =  1 - tl.cost.dice_coe(g_logits, real_images, axis=[0,1,2,3])
        g_vars = tl.layers.get_variables_with_name('u_net', True, True)


        g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(g_loss, var_list=g_vars)




    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    #saver.restore(sess, './checkpoint/model.ckpt')
    
    if resue_train==0:
        imgs = []
        for im in glob('D:/LIDC/xmlio/bmpdata/*.bmp'):
            imgs.append(im)
        imgs=np.asarray(imgs,np.chararray)
        ratio=0.9
        num_example=imgs.shape[0]
        arr=np.arange(num_example)
        np.random.shuffle(arr)
        imgs=imgs[arr]
        s=np.int(num_example*ratio)
        origin_files=imgs[:s]
        seg_files=imgs[s:]
       
    else:
        origin_files=np.load('./features2/origin_files.npy')
        seg_files=np.load('./features2/seg_files.npy')
        saver.restore(sess, './checkpoint/model.ckpt')
        
        

    ##========================= TRAIN MODELS ================================##
    iter_counter = 0
    for epoch in range(FLAGS.epoch):
        ## shuffle data
        shuffle(origin_files)
        shuffle(seg_files)
        ## update sample files based on shuffled data
        sample_files = seg_files[0:FLAGS.batch_size]
        sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        sample_a = sample_images[:,:,:,0]
        sample_a = (sample_a+1)/2
        sample_a=sample_a[:,:,:,np.newaxis]
        sample_b = sample_images[:,:,:,1]
        sample_b = (sample_b+1)/2
        sample_b=sample_b[:,:,:,np.newaxis]
        print("[*] Sample images updated!")

        ## load image data
        batch_idxs = min(len(origin_files), FLAGS.train_size) // FLAGS.batch_size
        valid_idxs = min(len(seg_files), FLAGS.train_size) // FLAGS.batch_size
        for idx in range(0, batch_idxs):
            start_time = time.time()
            batch_files = origin_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
            
            ## get real images
            batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            sample_seed = np.array(batch_images).astype(np.float32)
            batch_a = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,0]
            batch_a=batch_a[:,:,:,np.newaxis]
            batch_a = (batch_a+1)/2
            batch_b = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,1]
            batch_b = (batch_b+1)/2
            batch_b=batch_b[:,:,:,np.newaxis]


            for _ in range(1):
                errG, _ = sess.run([ g_loss, g_optim], feed_dict={z: batch_a, real_images: batch_b})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time,  errG))
        
            iter_counter += 1
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                img, errG = sess.run([g_logits,  g_loss], feed_dict={z : sample_a, real_images: sample_b})
                img = np.array(img).astype(np.float32)
                img = np.concatenate((sample_a,img,np.zeros((FLAGS.batch_size,FLAGS.output_size,FLAGS.output_size,1))), axis = 3)
                tl.visualize.save_images(img, [2,2], './{}/img_{:02d}_{:04d}.bmp'.format(FLAGS.sample_dir, epoch, idx))
                tl.visualize.save_images(sample_images, [2,2], './{}/origin_{:02d}_{:04d}.bmp'.format(FLAGS.sample_dir, epoch, idx))
                            
                if epoch >15:
                        ## get real images
                    for v_idx in range(0,valid_idxs):
                        batch_files = seg_files[v_idx*FLAGS.batch_size:(v_idx+1)*FLAGS.batch_size]
                        batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
                        batch_images = np.array(batch).astype(np.float32)
                        sample_seed = np.array(batch_images).astype(np.float32)
                        batch_a = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,0]
                        batch_a=batch_a[:,:,:,np.newaxis]
                        batch_a = (batch_a+1)/2
                        batch_b = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,1]
                        batch_b = (batch_b+1)/2
                        batch_b=batch_b[:,:,:,np.newaxis]
                        img, errG = sess.run([g_logits,  g_loss], feed_dict={z : batch_a, real_images: batch_b})
                        img = np.array(img).astype(np.float32)
                        save_name='%s%d%s%d%s' % ('./features2/features_',epoch,'_',idx,'.mat')
                        if v_idx == 0:
                            seg_img=img
                            ori_img=batch_a
                            label_img=batch_b
                        else:
                            seg_img=np.concatenate((seg_img,img),axis=3)
                            ori_img=np.concatenate((ori_img,batch_a),axis=3)
                            label_img=np.concatenate((label_img,batch_b),axis=3)
                        
                        print(v_idx)
                    scipy.io.savemat(save_name, mdict={'seg_img':seg_img,'ori_img':ori_img,'label_img':label_img})
                    np.save('./features2/origin_files.npy',origin_files)
                    np.save('./features2/seg_files.npy',seg_files)
                    saver.save(sess,'./checkpoint/model.ckpt')


        
if __name__ == '__main__':
    tf.app.run()
