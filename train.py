from keras.datasets import cifar10
from model import *
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

(trainX, _), (testX, _) = cifar10.load_data()

def normalize_pixels(train_data, test_data):
    #convert integer values to float
	train_norm = train_data.astype('float32')
	test_norm = test_data.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	return train_norm, test_norm


def train(model_str, model_f, compression_ratios, snr=10, nb_epoch=1, batch_size=1):

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

        x_train, x_test = normalize_pixels(trainX, testX)
        model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                  callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))



#------------------------------------------
model_str = 'model1'
model_f = model1
compression_ratios = [0.06, 0.09, 0.17]

train(model_str, model_f, compression_ratios)