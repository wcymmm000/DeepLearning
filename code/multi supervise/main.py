import os, sys, pprint, time
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from model2 import *
from model import *
import scipy.io


flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.00002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam ")
flags.DEFINE_integer("cmap_radius", 21, "Center map gaussian variance")
flags.DEFINE_integer("train_size", np.inf, "The size of train images ")
flags.DEFINE_integer("batch_size", 16, "The number of batch images ")
flags.DEFINE_integer("image_size", 512, "The size of image to use (will be center cropped) ")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce ")
flags.DEFINE_integer("sample_size", 16, "The number of sample images ")
flags.DEFINE_integer("sample_step", 592, "The interval of generating sample. ")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints ")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples ")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing ")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing ")
FLAGS = flags.FLAGS


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

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
    

    tl.files.exists_or_mkdir(FLAGS.checkpoint_dir)
    tl.files.exists_or_mkdir(FLAGS.sample_dir)
    with tf.device("/gpu:0"):
        
        input_img = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,1], name='input_img')
        label_heat =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,1], name='label_heat')
        label_img =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size,1], name='label_img')
		
        # z --> generator for training
        logits1 = heatmap_g(input_img)
        logits2 = unet(input_img, logits1)

        loss1 =  tl.cost.mean_squared_error(logits1, label_heat, is_mean=False, name='mean_squared_error')
        loss2 =  1 - tl.cost.dice_coe(logits2, label_img, axis=[0,1,2,3])
        #loss = tl.cost.sigmoid_cross_entropy(logits, real_images, name='gfake')
        #loss = tl.cost.iou_coe(logits, real_images, axis=[0,1,2])
        #loss = tl.cost.mean_squared_error(logits, label_img, is_mean=False, name='mean_squared_error')
        vars1 = tl.layers.get_variables_with_name('CPM', True, True)
        vars2 = tl.layers.get_variables_with_name('u_net', True, True)

        optim1 = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(loss1, var_list=vars1)
        optim2 = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                          .minimize(loss2, var_list=vars2)           

    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)    
    imgs = []
    for im in glob('D:/LIDC/xmlio/gauss/*.bmp'):
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
    

    

    ## Run Epoch
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
        sample_c = sample_images[:,:,:,2]
        sample_c = (sample_c+1)/2
        sample_c=sample_c[:,:,:,np.newaxis]
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
            batch_c = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,2]
            batch_c = (batch_c+1)/2
            batch_c=batch_c[:,:,:,np.newaxis]
            ## train model
            t_heat, errG1, _ = sess.run([logits1, loss1, optim1], feed_dict={input_img: batch_a, label_heat: batch_b})
            t_img, errG2, _ = sess.run([logits2, loss2, optim2], feed_dict={input_img: batch_a, label_img: batch_c})
            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, loss: %.8f" \
                % (epoch, FLAGS.epoch, idx, batch_idxs, time.time() - start_time,  errG2))
            ## valid and save
            

            iter_counter += 1
            if np.mod(iter_counter, 10) == 0:
                
                t_img = np.array(t_img).astype(np.float32)
                #print(t_img.shape)
                #print(batch_a.shape)
                t_img = np.concatenate((batch_a,t_img,np.zeros((FLAGS.batch_size,FLAGS.output_size,FLAGS.output_size,1))), axis = 3)
                tl.visualize.save_images(t_img, [4,4], './train_img/t_img{:02d}_{:04d}.bmp'.format(epoch, idx))
            
            if np.mod(iter_counter, FLAGS.sample_step) == 0:
                img, errG = sess.run([logits2,  loss2], feed_dict={input_img : sample_a, label_img: sample_c})
                print('-----0')
                print(sample_a.shape)
                print(type(sample_a))
                img = np.array(img).astype(np.float32)
                
                img = np.concatenate((sample_a,img,np.zeros((FLAGS.batch_size,FLAGS.output_size,FLAGS.output_size,1))), axis = 3)
                o_img = np.concatenate((sample_a,sample_c,np.zeros((FLAGS.batch_size,FLAGS.output_size,FLAGS.output_size,1))), axis = 3)
                
                tl.visualize.save_images(img, [4,4], './samples/img_{:02d}_{:04d}.bmp'.format(epoch, idx))
                tl.visualize.save_images(o_img, [4,4], './samples/origin_{:02d}_{:04d}.bmp'.format(epoch, idx))

                       
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
                        batch_c = sample_seed[0:FLAGS.batch_size, 0:FLAGS.output_size, 0:FLAGS.output_size,2]
                        batch_c = (batch_c+1)/2
                        batch_c=batch_c[:,:,:,np.newaxis]                        
                        t_img, t_errG = sess.run([logits2,  loss2], feed_dict={input_img : batch_a, label_img: batch_c})
                        t_img = np.array(t_img).astype(np.float32)
                        save_name='%s%d%s%d%s' % ('./features2/features_',epoch,'_',idx,'.mat')
                        if v_idx == 0:
                            seg_img=t_img
                            ori_img=batch_a
                            l_img=batch_c
                        else:
                            seg_img=np.concatenate((seg_img,t_img),axis=3)
                            ori_img=np.concatenate((ori_img,batch_a),axis=3)
                            l_img=np.concatenate((l_img,batch_c),axis=3)
                        
                    scipy.io.savemat(save_name, mdict={'seg_img':seg_img,'ori_img':ori_img,'label_img':l_img})
                    np.save('./features2/origin_files.npy',origin_files)
                    np.save('./features2/seg_files.npy',seg_files)
                    saver.save(sess,'./checkpoint/model.ckpt')


        
if __name__ == '__main__':
    tf.app.run()
