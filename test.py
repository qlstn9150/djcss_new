from model import normalize_pixels

import tensorflow as tf
from keras.models import load_model
from model import NormalizationNoise
from keras.datasets import cifar10
from keras import backend as K
from skimage.metrics import structural_similarity
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json

# 데이터 준비
(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

'''
# test
def comp_eval(model_str, x_test, compression_ratios, snr):
    model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
    model_dic['SNR'].append(snr)

    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        path = './checkpoints/' + model_str + '_CompRatio{0}_SNR{1}.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)

        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, multichannel=True)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Pred_Images'].append(pred_images)
        model_dic['PSNR'].append(psnr)
        model_dic['SSIM'].append(ssim)
        print('Comp_Ratio = ', comp_ratio)
        print('PSNR = ', psnr)
        print('SSIM = ', ssim)
        print('\n')

        path = './compratio_txt/' + model_str + '_CompRatio{0}_SNR{1}.txt'.format(comp_ratio, snr)
        with open(path, 'w') as f:
            print(compression_ratios, '\n', model_dic['PSNR'], file=f)
        f.closed
    return model_dic

'''
def test_eval(model_str, x_test, comp_ratio, snr, snr_test):
    model_dic = {'Test_snr': [], 'PSNR': []}
    for snr_t in snr_test:
        tf.keras.backend.clear_session()
        path = './checkpoints/' + model_str + '_CompRatio{0}_SNR{1}.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr_t)

        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Test_snr'].append(snr_t)
        model_dic['PSNR'].append(psnr)
        print('Test SNR = ', snr_t)
        print('PSNR = ', psnr)
        print('\n')

        path = './test_txt/' + model_str + '_CompRatio{0}_SNR{1}.txt'.format(comp_ratio, snr)
        with open(path, 'w') as f:
            print(snr_test, '\n', model_dic['PSNR'], file=f)
        f.closed

    return model_dic

'''
# plot
def comp_plot(model_str, x_test, compression_ratios, snr_train):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = list(mcolors.TABLEAU_COLORS)
    history = []

    for model in model_str:
        i = 0
        for snr in snr_train:
            print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
            model_dic = comp_eval(model, x_test, compression_ratios, snr) #mode='multiple'
            history.append(model_dic)
            label = model + ' (SNR={0}dB)'.format(snr)
            plt.plot(compression_ratios, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
            i += 1

            plt.title('AWGN Channel')
            plt.xlabel('k/n')
            plt.ylabel('PSNR (dB)')
            plt.grid(True)

        #plt.ylim(11, 21)
        plt.legend(loc='lower right')
        plt.show()
'''
def test_plot(model_str, x_test, comp_ratio, snr_train, snr_test):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = list(mcolors.TABLEAU_COLORS)
    history = []
    for model in model_str:
        i = 0
        for snr in snr_train:
            print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
            model_dic = test_eval(model, x_test, comp_ratio, snr, snr_test)
            history.append(model_dic)
            label = model + ' (SNR_train={0}dB)'.format(snr)
            plt.plot(snr_test, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
            i += 1

            plt.title('AWGN Channel')
            plt.xlabel('SNR_test (dB)')
            plt.ylabel('PSNR (dB)')
            plt.grid(True)

        plt.ylim(11, 21)
        plt.legend(loc='lower right')
        plt.show()



### 실행
#plot1
model_str = ['model1']
'''
compression_ratios = [0.06] #0.26, 0.49
snr_train = [10] #0, 10, 20

comp_plot(model_str, x_test, compression_ratios, snr_train)

'''
#plot2
comp_ratio = 0.06
snr_train = [10]
snr_test = [2, 4, 7, 10, 13, 16, 18, 22, 25, 27]

test_plot(model_str, x_test, comp_ratio, snr_train, snr_test)
