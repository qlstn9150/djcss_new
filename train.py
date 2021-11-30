import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from model import *
from model2 import *

import tensorflow as tf
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import time

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

def train(model_str, model_f, all_snr, compression_ratios, nb_epoch, batch_size=16):
    for snr in all_snr:
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()

            model = model_f(comp_ratio)
            model.summary()

            K.set_value(model.get_layer('normalization_noise_1').snr_db, snr)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

            os.makedirs('./Tensorboard/{0}'.format(model_str), exist_ok=True)
            os.makedirs('./checkpoints/{0}'.format(model_str), exist_ok=True)

            tb = TensorBoard(log_dir='./Tensorboard/{0}/CompRatio{1}_SNR{2}'.format(model_str, str(comp_ratio), str(snr)))

            checkpoint = ModelCheckpoint(filepath='./checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model_str, str(comp_ratio), str(snr)),
                                         monitor = 'val_loss', save_best_only = True)

            ckpt = ModelCheckponitsHandler(model_str, comp_ratio, snr, model, step=50)


            start = time.clock()
            model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                      callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
            end = time.clock()
            print('The NN has trained ' + str(end - start) + ' s')


#------------------------------------------
model_str = 'basic750'
model_f = basic750
all_snr = [0,10,20]
compression_ratios = [0.06, 0.26, 0.49]

nb_epoch = 750

train(model_str, model_f, all_snr, compression_ratios, nb_epoch)