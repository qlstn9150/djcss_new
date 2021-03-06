'''
tensorflow: v1.15.0
keras: v2.3.1
'''

from tensorflow.python.client import device_lib
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device_lib.list_local_devices()

import tensorflow as tf
tf.config.list_physical_devices('GPU')

from keras.datasets import cifar10

from train_eval import *
from plot import *

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

model = 'model14'
model_f = model14
model_list = ['basic', model] #compare
snr_train = [0, 20]
snr_test = [2, 10, 18, 26] #2, 4, 7, 10, 13, 16, 18, 22, 25, 27
compression_ratios = [0.06, 0.26, 0.49]

#check
train(model, model_f, snr_train, compression_ratios, x_train, x_test, nb_epoch=5) #750
#comp_eval(model, compression_ratios, snr_train, x_test, testX)
#comp_psnr_plot(model_list, snr_train)

#compare
model_list = ['basic', 'model8', model]
all_model_psnr(model_list, snr_train)

#if good
#comp_ssim_plot(model_list, snr_train)
#test_eval(model, x_test, compression_ratios, snr_train, snr_test, testX)
#test_plot(model_list, compression_ratios, snr_train)
#save_img(model)
