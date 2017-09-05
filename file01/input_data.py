# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 20:31:16 2017

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import os

#%%
img_width = 208
img_height = 208

#%%
#train_dir = 'D:/anaconda/dailycode/catsVSdogs/catvsdogs/data/train/'
def get_files(file_dir):#返回存放数据的路径以及label标签
    '''
    args:
        file_dir: file directory
    returns:
        list of images and labels
    '''
    cats =  []
    label_cats = []
    dogs = []
    label_dogs = []
    for file in os.listdir(file_dir):#返回这个路径下所有文件的名字
        name = file.split(sep='.')
        if name[0]=='cat':
            cats.append(file_dir +file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print('there are %d cats\nthere are %d dogs' %(len(cats),len(dogs)))
            
    image_list = np.hstack((cats, dogs))#将猫狗图片堆积起来
    label_list = np.hstack((label_cats,label_dogs))#label也堆积起来
    
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)#打乱数据
    
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
 
    
    return image_list,label_list

#%%
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    '''
    Args:
        image: list type get_files函数返回的image_list
        label: list type  get_files函数返回的label_list
        image_W: image width 图片大小不一，需要裁减的宽高
        image_H: image width
        batch_size: batch size，每一批的图片数量
        capacity: the maximum elements in queue队列中容纳的图片数
    Returns:
        iamge_batch:4D tensor [batch_size,width,height,3],dtype=tf.float32；#图像是rgb所以通道是3
        label_batch:1D tensor [batch_size],dtype=tf.int32
    
    '''
    global label_batch
    #在python的函数中和全局同名的变量，如果你有修改变量的值就会变成局部变量，
    #在修改之前对该变量的引用自然就会出现没定义这样的错误了，如果确定要引用全局变量，并且要对它修改，必须加上global关键字。
    image = tf.cast(image,tf.string) #转换数据类型
    label = tf.cast(label,tf.int32)
    
    #make  an input queue输入队列
    input_queue = tf.train.slice_input_producer([image,label])
    
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)#解码
    ###############################
    #data argymentation should go to here可以做一些数据特征加强
    #################################
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)#裁减图片的长宽
    image = tf.image.per_image_standardization(image)#神经网络对图片要求很高，所以标准化图片减去均值除以方差
    image_batch,label_batch = tf.train.batch([image,label],#list
                                           batch_size = batch_size,
                                           num_threads = 64,
                                           capacity = capacity)#队列中最多能容纳的个数
 #image_batch,label_batch = tf.train.shuffle_batch([image,label],
 #                                                 batch_size = BATCH_SIZE,
 #                                                 num_threads=64,
 #                                                 capacity=CAPACITY,
 #                                                 min_after_dequeue=CAPACITY-1)  
    label_batch = tf.reshape(label_batch,[batch_size])
    
    return image_batch,label_batch


#%%TEST those two function

import matplotlib.pyplot as plt

BATCH_SIZE = 10
CAPACITY = 256 #队列中最多容纳图片的个数
IMG_W = 208
IMG_H = 208
train_dir = 'D:/anaconda/dailycode/catsVSdogs/catsvsdogs/data/train/'
image_list,label_list = get_files(train_dir)
image_batch,label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    try:
        while not coord.should_stop() and i<1:
            img,label = sess.run([image_batch,label_batch])
            
            #just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])#4D数据后面全用冒号
                plt.show()
            i+=1
            
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)




                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
                
    
    
    
    