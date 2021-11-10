'''
tensorflow: v1.15.0
keras: v2.3.1
'''
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

from keras.layers import Conv2D, Layer, Input, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.layers import Add, Dense, BatchNormalization, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.models import Model

class NormalizationNoise(Layer):
    def __init__(self, snr_db_def=20, P_def=1, name='NormalizationNoise', **kwargs):
        self.snr_db = K.variable(snr_db_def, name='SNR_db')
        self.P = K.variable(P_def, name='Power')
        self.name = name
        super(NormalizationNoise, self).__init__(**kwargs)

    def call(self, z_tilta):
        with tf.name_scope('Normalization_Layer'):
            z_tilta = tf.dtypes.cast(z_tilta, dtype='complex128', name='ComplexCasting') + 1j
            lst = z_tilta.get_shape().as_list()
            lst.pop(0)
            # computing channel dimension 'k' as the channel bandwidth.
            k = np.prod(lst, dtype='float32')
            # calculating conjugate transpose of z_tilta
            z_conjugateT = tf.math.conj(tf.transpose(z_tilta, perm=[0, 2, 1, 3], name='transpose'),
                                        name='z_ConjugateTrans')
            # Square root of k and P
            sqrt1 = tf.dtypes.cast(tf.math.sqrt(k * self.P, name='NormSqrt1'), dtype='complex128',
                                   name='ComplexCastingNorm')
            sqrt2 = tf.math.sqrt(z_conjugateT * z_tilta, name='NormSqrt2')  # Square root of z_tilta* and z_tilta.

            div = tf.math.divide(z_tilta, sqrt2, name='NormDivision')
            # calculating channel input
            z = tf.math.multiply(sqrt1, div, name='Z')

        with tf.name_scope('PowerConstraint'):
            ############# Implementing Power Constraint ##############
            z_star = tf.math.conj(tf.transpose(z, perm=[0, 2, 1, 3], name='transpose_Pwr'), name='z_star')
            prod = z_star * z
            real_prod = tf.dtypes.cast(prod, dtype='float32', name='RealCastingPwr')
            pwr = tf.math.reduce_mean(real_prod)
            cmplx_pwr = tf.dtypes.cast(pwr, dtype='complex128', name='PowerComplexCasting') + 1j
            pwr_constant = tf.constant(1.0, name='PowerConstant')
            # Z: satisfies 1/kE[z*z] <= P, where P=1
            Z = tf.cond(pwr > pwr_constant, lambda: tf.math.divide(z, cmplx_pwr), lambda: z, name='Z_fixed')
            #### AWGN ###

        with tf.name_scope('AWGN_Layer'):
            k = k.astype('float64')
            # Converting SNR from db scale to linear scale
            snr = 10 ** (self.snr_db / 10.0)
            snr = tf.dtypes.cast(snr, dtype='float64', name='Float32_64Cast')
            ########### Calculating signal power ###########
            # calculate absolute value of input
            abs_val = tf.math.abs(Z, name='abs_val')
            # Compute Square of all values and after that perform summation
            summation = tf.math.reduce_sum(tf.math.square(abs_val, name='sq_awgn'), name='Summation')
            # Computing signal power, dividing summantion by total number of values/symbols in a signal.
            sig_pwr = tf.math.divide(summation, k, name='Signal_Pwr')
            # Computing Noise power by dividing signal power by SNR.
            noise_pwr = tf.math.divide(sig_pwr, snr, name='Noise_Pwr')
            # Computing sigma for noise by taking sqrt of noise power and divide by two because our system is complex.
            noise_sigma = tf.math.sqrt(noise_pwr / 2, name='Noise_Sigma')

            # creating the complex normal distribution.
            z_img = tf.math.imag(Z, name='Z_imag')
            z_real = tf.math.real(Z, name='Z_real')
            rand_dist = tf.random.normal(tf.shape(z_real), dtype=tf.dtypes.float64, name='RandNormalDist')
            # Compute product of sigma and complex normal distribution
            noise = tf.math.multiply(noise_sigma, rand_dist, name='Noise')
            # adding the awgn noise to the signal, noisy signal: ẑ
            z_cap_Imag = tf.math.add(z_img, noise, name='z_cap_Imag')
            z_cap_Imag = tf.dtypes.cast(z_cap_Imag, dtype='float32', name='NoisySignal_Imag')

            z_cap_Real = tf.math.add(z_real, noise, name='z_cap_Real')
            z_cap_Real = tf.dtypes.cast(z_cap_Real, dtype='float32', name='NoisySignal_Real')

            return z_cap_Real

class ModelCheckponitsHandler(tf.keras.callbacks.Callback):
    def __init__(self, model_str, comp_ratio, snr_db, autoencoder, step):
        super(ModelCheckponitsHandler, self).__init__()
        self.model_str = model_str
        self.comp_ratio = comp_ratio
        self.snr_db = snr_db
        self.step = step
        self.autoencoder = autoencoder

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.step == 0:
            os.makedirs('./CKPT_ByEpochs/' + str(self.model_str) + '_CompRatio' + str(self.comp_ratio) +
                        '_SNR' + str(self.snr_db), exist_ok=True)
            path = './CKPT_ByEpochs/' + str(self.model_str) + '_CompRatio' + str(self.comp_ratio) + \
                   '_SNR' + str(self.snr_db) + '/Epoch_' + str(epoch) + '.h5'
            self.autoencoder.save(path)
            print('\nModel Saved After {0} epochs.'.format(epoch))


def Calculate_filters(comp_ratio, F=5, n=3072):
    K = (comp_ratio * n) / F ** 2
    return int(K)

def model1(c):
    num_fiters_1 = 128
    kernel_size_1 = (32,32)
    kernel_size_2 = (4,4)
    kernel_size_3 = (1,1)

    input = Input(shape=(32, 32, 3))
    e = Dense(c)(input)

    noise = NormalizationNoise()(e)

    # Define Decoder Layers (Receiver)
    d1 = Conv2D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_1, padding='same')(noise)
    d1 = BatchNormalization()(d1)
    d1 = Activation('elu')(d1)

    d2 = Conv2D(filters=num_fiters_1, strides=1, kernel_size=kernel_size_2, padding='same')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Activation('elu')(d2)

    d2 = Add()([d1, d2])

    output = Conv2D(filters=3, strides=1, kernel_size=kernel_size_3, padding='same', activation='softmax')(d2)

    model = Model(inputs=input, outputs=output)
    #model.summary()
    return model