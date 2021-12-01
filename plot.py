import os
import tensorflow as tf
from keras.datasets import cifar10
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import array_to_img

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
import glob
import cv2

from model import normalize_pixels
from model import NormalizationNoise


def comp_psnr_plot(model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    markers = ['o', '*', 'H']
    ls = ['--', '-']
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
            plt.plot(compression_ratios, psnr, ls=ls[i], c=colors[j], marker=markers[i], label=label)
            #plt.plot(compression_ratios, psnr, ls='-', c=colors[i], marker='o', label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel')
    plt.xlabel('k/n')
    plt.ylabel('PSNR (dB)')
    plt.ylim(0, 35)
    plt.grid(True)
    plt.legend(loc='lower right')
    os.makedirs('./plot/plot1_psnr', exist_ok=True)
    plt.savefig('./plot/plot1_psnr/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, compression_ratios, snr_train))
    plt.show()

def comp_ssim_plot(model_str, snr_train):
    colors = list(mcolors.TABLEAU_COLORS)
    markers = ['o', '*', 'H']
    ls = ['--', '-']
    i = 0
    for model in model_str:
        j = 0
        for snr in snr_train:
            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'r') as f:
                text = f.read()
                compression_ratios = text.split('\n')[0]
                compression_ratios = json.loads(compression_ratios)
                ssim = text.split('\n')[2]
                ssim = json.loads(ssim)
            label = '{0} (SNR={1}dB)'.format(model, snr)
            plt.plot(compression_ratios, ssim, ls=ls[i], c=colors[j], marker=markers[i], label=label)
            #plt.plot(compression_ratios, ssim, ls='-', c=colors[i], marker='X', label=label)
            j += 1
        i += 1
    plt.title('AWGN Channel')
    plt.xlabel('k/n')
    plt.ylabel('SSIM')
    plt.ylim(0.4,1)
    plt.grid(True)
    plt.legend(loc='lower right')
    os.makedirs('./plot/plot1_ssim', exist_ok=True)
    plt.savefig('./plot/plot1_ssim/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, compression_ratios, snr_train))
    plt.show()

def test_plot(model_str, compression_ratios, snr_train):
    for comp_ratio in compression_ratios:
        colors = list(mcolors.TABLEAU_COLORS)
        markers = ['s', 'H', '^']
        i = 0
        #markers = ['o', '*', 'H']
        ls = ['--', '-']
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
                plt.plot(snr_test, psnr, ls=ls[i], c=colors[j], marker=markers[i], label=label)
                j += 1
            i += 1
        plt.title('AWGN Channel (k/n={0})'.format(comp_ratio))
        plt.xlabel('SNR_test (dB)')
        plt.ylabel('PSNR (dB)')
        plt.ylim(0,35)
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.savefig('./plot/plot2/{0}_CompRatio{1}_SNR{2}.png'.format(model_str, comp_ratio, snr_train))
        plt.show()


def save_img(model, compression_ratios, snr_train):
    (trainX, _), (testX, _) = cifar10.load_data()
    _, x_test = normalize_pixels(trainX, testX)
    original_images = array_to_img(x_test[0,])
    original_images.save('./img/original.jpg')

    os.makedirs('./img/{0}'.format(model), exist_ok=True)
    for snr in snr_train:
        for comp_ratio in compression_ratios:
            tf.keras.backend.clear_session()
            print('==============={0}_CompRation{1}_SNR{2}============'.format(model, comp_ratio, snr))
            path = './checkpoints/{0}/CompRatio{1}_SNR{2}.h5'.format(model, comp_ratio, snr)
            autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})

            #input image
            #e_output = e_output.predict(x_test) * 255
            #e_output = array_to_img(e_output .astype('uint8'))

            #encoder output
            '''e_output = autoencoder.get_layer('e_output').output * 255
            e_output = e_output.astype('uint8')
            e_output = array_to_img(e_output[6,])
            e_output.save('./img/{0}/eoutput_CompRatio{1}_SNR{2}.jpg'.format(model, comp_ratio, snr))
'''
            #channel output
            K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
            #c_output = autoencoder.get_layer('c_output').output

            # decoder output
            pred_images = autoencoder.predict(x_test) * 255
            pred_images = pred_images.astype('uint8')
            pred_images = array_to_img(pred_images[6,])
            pred_images.save('./img/{0}/pred_CompRatio{1}_SNR{2}.jpg'.format(model, comp_ratio, snr))

    fig = plt.figure()
    i=1
    for filename in sorted(glob.glob('./img/{0}/*.jpg'.format(model))):
        print(filename)
        img = cv2.imread(filename)
        ax = fig.add_subplot(3,3,i)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax = plt.gca()
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

        label = filename.replace('./img/{0}/pred_'.format(model), '')
        label = label.replace('.jpg', '')
        ax.set_xlabel(label)
        i += 1
    plt.save('./img/{0}/all.jpg'.format(model))
    plt.show()