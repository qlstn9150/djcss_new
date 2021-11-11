from model import *
import tensorflow as tf
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import time

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

def train(model_str, model_f, compression_ratios, nb_epoch, snr=10, batch_size=16):

    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()

        c = Calculate_filters(comp_ratio)
        print('---> System Will Train, Compression Ratio: '+str(comp_ratio)+'. <---')
        model = model_f(c)

        K.set_value(model.get_layer('normalization_noise_1').snr_db, snr)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

        tb = TensorBoard(log_dir='./Tensorboard/' + model_str + '_CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))

        # os.makedirs('./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
        checkpoint = ModelCheckpoint(filepath='./checkpoints/' + model_str + '_CompRatio{0}_SNR{1}.h5'.format(str(comp_ratio), str(snr)),
                                     monitor = 'val_loss', save_best_only = True)

        ckpt = ModelCheckponitsHandler(model_str, comp_ratio, snr, model, step=50)


        start = time.clock()
        model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                  callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))
        end = time.clock()
        print('The NN has trained ' + str(end - start) + ' s')


#------------------------------------------
model_str = 'model1'
model_f = model1
compression_ratios = [0.26, 0.49] #0.06
nb_epoch = 5

train(model_str, model_f, compression_ratios, nb_epoch)