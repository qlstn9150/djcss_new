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

#normalizing the training and test data
x_train, x_test = normalize_pixels(trainX, testX)
#compression_ratios = [0.06, 0.09, 0.17, 0.26, 0.34, 0.43, 0.49]
compression_ratios = [0.06, 0.09, 0.17]
SNR = [0, 10, 20]

for comp_ratio in compression_ratios:
    tf.keras.backend.clear_session()
    c = Calculate_filters(comp_ratio)
    print('---> System Will Train, Compression Ratio: '+str(comp_ratio)+'. <---')

    model = model1(x_train, x_test, nb_epoch=750, comp_ratio=comp_ratio, batch_size=16, c=c, snr=10, saver_step=50)

    K.set_value(model.get_layer('normalization_noise_1').snr_db, snr)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])


    print('\t-----------------------------------------------------------------')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t| Training Parameters: Filter Size: {0}, Compression ratio: {1} |'.format(c, comp_ratio))
    print('\t|\t\t\t  SNR: {0} dB\t\t\t\t|'.format(snr))
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t|\t\t\t\t\t\t\t\t|')
    print('\t-----------------------------------------------------------------')

    tb = TensorBoard(log_dir='./Tensorboard_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)))
    # os.makedirs('./checkpoints_DN/CompRatio{0}_SNR{1}'.format(str(comp_ratio), str(snr)), exist_ok=True)
    checkpoint = ModelCheckpoint(filepath='./checkpoints/CompRatio{0}_SNR{1}.h5'.format(str(comp_ratio), str(snr)),
    monitor = 'val_loss', save_best_only = True)
    ckpt = ModelCheckponitsHandler(comp_ratio, snr, model, step=saver_step)
    model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=nb_epoch,
                        callbacks=[tb, checkpoint, ckpt], validation_data=(x_test, x_test))