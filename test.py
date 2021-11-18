'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
'''

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


def comp_eval(model_str, x_test, compression_ratios, snr_train):
    for model in model_str:
        for snr in snr_train:
            model_dic = {'Pred_Images': [], 'PSNR': [], 'SSIM': []}
            for comp_ratio in compression_ratios:
                tf.keras.backend.clear_session()
                path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
                autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
                K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)

                pred_images = autoencoder.predict(x_test) * 255
                pred_images = pred_images.astype('uint8')
                #ssim = structural_similarity(testX, pred_images, multichannel=True)
                psnr = peak_signal_noise_ratio(testX, pred_images)
                model_dic['Pred_Images'].append(pred_images)
                model_dic['PSNR'].append(psnr)
                #model_dic['SSIM'].append(ssim)
                print('Comp_Ratio = ', comp_ratio)
                print('PSNR = ', psnr)
                #print('SSIM = ', ssim)
                print('\n')

            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'w') as f:
                print(compression_ratios, '\n', model_dic['PSNR'], file=f)
            f.closed


def comp_plot(model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    markers = ['o', '*', 'H']
    #ls = ['-', '--']
    i = 0
    for model in model_str:
        j = 0
        for snr in snr_train:
            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'r') as f:
                text = f.read()
                compression_ratios = text.split('\n')[0]
                compression_ratios = json.loads(compression_ratios)
                psnr = text.split('\n')[1]
                psnr = json.loads(psnr)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(compression_ratios, psnr, ls='-', c=colors[i], marker=markers[i], label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel')
    plt.xlabel('k/n')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig('./plot/plot1/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, compression_ratios, snr_train))
    plt.show()



def test_eval(model_str, x_test, comp_ratio, snr_train, snr_test):
    for model in model_str:
        for snr in snr_train:
            model_dic = {'Test_snr': [], 'PSNR': []}
            for snr_t in snr_test:
                tf.keras.backend.clear_session()
                path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
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

            path = './result_txt/plot2/{0}_CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
            with open(path, 'w') as f:
                print(snr_test, '\n', model_dic['PSNR'], file=f)
            f.closed


def test_plot(model_str, comp_ratio, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    markers = ['s', 'H', '^']
    i = 0
    for model in model_str:
        j = 0
        for snr in snr_train:
            path = './result_txt/plot2/{0}_CompRatio{1}_SNR{2}.txt'.format(model, comp_ratio, snr)
            with open(path, 'r') as f:
                text = f.read()
                snr_test = text.split('\n')[0]
                snr_test = json.loads(snr_test)
                psnr = text.split('\n')[1]
                psnr = json.loads(psnr)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(snr_test, psnr, ls='--', c=colors[i], marker=markers[j], label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel (k/n={0})'.format(comp_ratio))
    plt.xlabel('SNR_test (dB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig('./plot/plot2/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, comp_ratio, snr_train))
    plt.show()



#실행

#===========plot1================
model_str = ['basic', 'model2', 'new2']
compression_ratios = [0.06, 0.26, 0.49] #0.26, 0.49
snr_train = [0, 10, 20] #0, 10, 20
#comp_eval(model_str, x_test, compression_ratios, snr_train)
#comp_plot(model_str, snr_train)

#===========plot2================
model_str = ['new2']
comp_ratio = 0.49 #0.06, 0.26, 0.49
snr_train = [0, 10, 20]
snr_test = [2, 10, 18, 26] #2, 4, 7, 10, 13, 16, 18, 22, 25, 27
#test_eval(model_str, x_test, comp_ratio, snr_train, snr_test)
#test_plot(model_str, comp_ratio, snr_train)



