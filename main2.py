'''import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
'''

from train2 import *
from test import *
from plot import *

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
tf.config.list_physical_devices('GPU')

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

model = 'model'
model_f = jpeg
model_str = ['basic', 'jpeg'] #compare
snr_train = [0,10,20]
snr_test = [2, 10, 18, 26] #2, 4, 7, 10, 13, 16, 18, 22, 25, 27
compression_ratios = [0.06, 0.26, 0.49]

### TRAIN
train(model, model_f, snr_train, compression_ratios, nb_epoch=5) #750

### TEST
comp_eval(model, x_test, compression_ratios, snr_train)
test_eval(model, x_test, compression_ratios, snr_train, snr_test)

### PLOT
# image
#save_img(model, compression_ratios, snr_train)
# plot 1
comp_psnr_plot(model_str, snr_train)
#comp_ssim_plot(model_str, snr_train)
# plot 2
#test_plot(model_str, compression_ratios, snr_train)