import tensorflow as tf
from keras.datasets import cifar10
from model2 import normalize_pixels

(trainX, _), (testX, _) = cifar10.load_data()
x_train, x_test = normalize_pixels(trainX, testX)

jpeg = tf.image.encode_jpeg(x_train, format='rgb')
print(jpeg)

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
                ssim = structural_similarity(testX, pred_images, multichannel=True)
                psnr = peak_signal_noise_ratio(testX, pred_images)
                model_dic['Pred_Images'].append(pred_images)
                model_dic['PSNR'].append(psnr)
                model_dic['SSIM'].append(ssim)
                print('Comp_Ratio = ', comp_ratio)
                print('PSNR = ', psnr)
                print('SSIM = ', ssim)
                print('\n')

            path = './result_txt/plot1/{0}_SNR{1}.txt'.format(model, snr)
            with open(path, 'w') as f:
                print(compression_ratios, '\n', model_dic['PSNR'], '\n', model_dic['SSIM'], file=f)
            f.closed