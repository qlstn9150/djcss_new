import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json


import tensorflow as tf
from keras.models import load_model
from model import NormalizationNoise
from keras.datasets import cifar10
from keras import backend as K
from skimage.metrics import structural_similarity
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio

####################################################################################################
# 데이터 준비
(trainX, _), (testX, _) = cifar10.load_data()


def normalize_pixels(train_data, test_data):
    train_norm = train_data.astype('float32')
    test_norm = test_data.astype('float32')
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    return train_norm, test_norm


x_train, x_test = normalize_pixels(trainX, testX)


####################################################################################################
# Basic CNN
def comp_eval(x_test, compression_ratios, snr, mode='multiple'):
    model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
    model_dic['SNR'].append(snr)
    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, multichannel=True)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Pred_Images'].append(pred_images)
        model_dic['PSNR'].append(psnr)
        model_dic['SSIM'].append(ssim)
    return model_dic


# DenseNet
def comp_eval_DN(x_test, compression_ratios, snr, mode='multiple'):
    model_dic = {'SNR': [], 'Pred_Images': [], 'PSNR': [], 'SSIM': []}
    model_dic['SNR'].append(snr)
    for comp_ratio in compression_ratios:
        tf.keras.backend.clear_session()
        path = './checkpoints_DN/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        ssim = structural_similarity(testX, pred_images, multichannel=True)
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Pred_Images'].append(pred_images)
        model_dic['PSNR'].append(psnr)
        model_dic['SSIM'].append(ssim)
    return model_dic


# 그래프 생성
def comp_plot(x_test, compression_ratios, snr_lst, title, x_lablel, y_label):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00']
    history = []
    i = 0
    for snr in snr_lst:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = comp_eval(x_test, compression_ratios, snr, mode='multiple')
        history.append(model_dic)
        label = 'DJSCC (SNR={0}dB)'.format(snr)
        plt.plot(compression_ratios, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1

    i=0
    for snr in snr_lst:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = comp_eval_DN(x_test, compression_ratios, snr, mode='multiple')
        history.append(model_dic)
        label = 'DJSCC-DN (SNR={0}dB)'.format(snr)
        plt.plot(compression_ratios, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1
        plt.title(title)
        plt.xlabel(x_lablel)
        plt.ylabel(y_label)
        plt.grid(True)

    plt.ylim(11, 21)
    plt.legend(loc='lower right')
    plt.show()
    return history


# 실행
compression_ratios = [0.06, 0.26, 0.49]
snr_train = [0, 10, 20]

history = comp_plot(x_test, compression_ratios, snr_train, title='AWGN Channel', x_lablel='k/n', y_label='PSNR (dB)')


####################################################################################################
# Basic CNN
def snr_eval(x_test, comp_ratio, snr_test, snr_train):
    model_dic = {'Train_snr': [snr_train], 'Test_snr': [], 'PSNR': []}
    for snr in snr_test:
        tf.keras.backend.clear_session()
        path = './checkpoints/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr_train)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Test_snr'].append(snr)
        model_dic['PSNR'].append(psnr)
    return model_dic

# DenseNet
def snr_eval_DN(x_test, comp_ratio, snr_test, snr_train):
    model_dic = {'Train_snr': [snr_train], 'Test_snr': [], 'PSNR': []}
    for snr in snr_test:
        tf.keras.backend.clear_session()
        path = './checkpoints_DN/CompRatio{0}_SNR{1}/Autoencoder.h5'.format(comp_ratio, snr_train)
        autoencoder = load_model(path, custom_objects={'NormalizationNoise': NormalizationNoise})
        K.set_value(autoencoder.get_layer('normalization_noise_1').snr_db, snr)
        pred_images = autoencoder.predict(x_test) * 255
        pred_images = pred_images.astype('uint8')
        psnr = peak_signal_noise_ratio(testX, pred_images)
        model_dic['Test_snr'].append(snr)
        model_dic['PSNR'].append(psnr)
    return model_dic

# 그래프 생성
def snr_plot(x_test, compression_ratio, snr_train, title, x_lablel, y_label):
    markers = ["*", "s", "o", "X", "d", "v", "<", ">", "^", "P", "H", "|"]
    colors = ['#800080', '#FF00FF', '#000080', '#008080', '#00FFFF', '#008000', '#00FF00']
    history = []
    i = 0
    for snr in snr_train:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = snr_eval(x_test, compression_ratio, snr_test, snr)
        history.append(model_dic)
        label = 'DJSCC (SNR_train={0}dB)'.format(snr)
        plt.plot(snr_test, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1

    i=0
    for snr in snr_train:
        print('\n----> Now Getting Data and Preparing Plot for SNR {0} dB <----'.format(snr))
        model_dic = snr_eval_DN(x_test, compression_ratio, snr_test, snr)
        history.append(model_dic)
        label = 'DJSCC-DN (SNR_train={0}dB)'.format(snr)
        plt.plot(snr_test, model_dic['PSNR'], ls='--', c=colors[i], marker=markers[i], label=label)
        i += 1
        plt.title(title)
        plt.xlabel(x_lablel)
        plt.ylabel(y_label)
        plt.grid(True)

    plt.ylim(11, 21)
    plt.legend(loc='lower right')
    plt.show()
    return history


# 실행
snr_train = [0, 10, 20]
snr_test = [2, 4, 7, 10, 13, 16, 18, 22, 25, 27]
comp_ratio = 0.49
history2 = snr_plot(x_test, comp_ratio, snr_train, title='AWGN Channel', x_lablel='SNR_test (dB)', y_label='PSNR (dB)')



def plot(channel, model_list):
    color_list = list(mcolors.TABLEAU_COLORS)
    ber_list = []
    label_list = []

    if channel == 'awgn':
        path = './awgn/'
    elif channel == 'bursty':
        path = './bursty/'
    else:
        path = './rayleigh/'

    for name in model_list:
        with open(path + name + '.txt', 'r') as f:
            text = f.read()
            Vec_Eb_N0 = text.split('\n')[0]
            Vec_Eb_N0 = json.loads(Vec_Eb_N0)
            ber = text.split('\n')[1]
            ber = json.loads(ber)
        ber_list.append(ber)
        label_list.append(name)


    for i in range(len(model_list)):
        plt.semilogy(Vec_Eb_N0, ber_list[i], label=label_list[i], color=color_list[i], marker='o')

    plt.legend(loc=0)
    plt.xlabel('Eb/N0(dB)')
    plt.ylabel('BER')
    plt.title(channel + ' Channel')
    plt.grid('true')
    plt.show()

# RUN
model_list = ['basic', 'model5']
channel = 'bursty'
plot(channel, model_list)