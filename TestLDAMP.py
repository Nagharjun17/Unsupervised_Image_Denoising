import time
import numpy as np
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
from matplotlib import pyplot as plt
import LearnedDAMP as LDAMP
import random
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/TrainingData",
    # default="/Users/arjun17/Documents/D-AMP_Toolbox/LDAMP_TensorFlow/TrainingData",
    help="Dataset path")
parser.add_argument(
    "--model_path",
    type=str,
    default="/scratch/nm4074/imageprocessing/D-AMP_Toolbox/LDAMP_TensorFlow/saved_models/LDAMP",
    # default="/Users/arjun17/Documents/D-AMP_Toolbox/LDAMP_TensorFlow/saved_models/LDAMP",
    help="Saved model path")
FLAGS, unparsed = parser.parse_known_args()

height_img = 256
width_img = 256
channel_img = 1
filter_height = 3
filter_width = 3
num_filters = 64
n_DnCNN_layers=16
n_DAMP_layers=10

LayerbyLayer=True

BATCH_SIZE = 1
n_Test_Images = 7
sampling_rate_test=1
sampling_rate_train=.2
sigma_w=20./255.
n=channel_img*height_img*width_img
m=int(np.round(sampling_rate_test*n))
measurement_mode='Fast-JL'

init_mu = 0
init_sigma = 0.1

random.seed(1)

LDAMP.SetNetworkParams(new_height_img=height_img, new_width_img=width_img, new_channel_img=channel_img, \
                       new_filter_height=filter_height, new_filter_width=filter_width, new_num_filters=num_filters, \
                       new_n_DnCNN_layers=n_DnCNN_layers, new_n_DAMP_layers=n_DAMP_layers,
                       new_sampling_rate=sampling_rate_test, \
                       new_BATCH_SIZE=BATCH_SIZE, new_sigma_w=sigma_w, new_n=n, new_m=m, new_training=False, use_adaptive_weights=False)
LDAMP.ListNetworkParameters()

x_true = tf.placeholder(tf.float32, [n, BATCH_SIZE])

[A_handle, At_handle, A_val, A_val_tf]=LDAMP.GenerateMeasurementOperators(measurement_mode)

n_layers_trained = n_DAMP_layers
theta = [None] * n_layers_trained
for iter in range(n_layers_trained):
    with tf.variable_scope("Iter" + str(iter)):
        theta_thisIter = LDAMP.init_vars_DnCNN(init_mu, init_sigma)
    theta[iter] = theta_thisIter

y_measured= LDAMP.GenerateNoisyCSData_handles(x_true, A_handle, sigma_w, A_val_tf)
print("y_measured", y_measured.shape)
(x_hat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr, SSIM_history) = LDAMP.LDAMP(y_measured, A_handle, At_handle, A_val_tf, theta, x_true, tie=False, test=True)

if height_img>50:
    test_im_name = FLAGS.data_path + "/StandardTestData_" + str(height_img) + "Res.npy"
else:
    test_im_name = FLAGS.data_path + "/TestData_patch" + str(height_img) + ".npy"
test_images = np.load(test_im_name)
test_images=test_images[:,0,:,:]
assert (len(test_images)>=n_Test_Images), "Requested too much Test data"

x_test = np.transpose( np.reshape(test_images, (-1, height_img * width_img * channel_img)))
saver = tf.train.Saver() 
saver_dict={}

with tf.Session() as sess:
    save_name = LDAMP.GenLDAMPFilename(path = FLAGS.model_path, sampling_rate_override=sampling_rate_train) + ".ckpt"
    saver.restore(sess, save_name)

    print("Reconstructing Signal")
    start_time = time.time()

    Final_PSNRs=[]
    Final_SSIMs=[]
    recon = []
    currentFinalPSNR = []
    currentFinalSSIM = []
    for offset in range(0, n_Test_Images - BATCH_SIZE + 1, BATCH_SIZE):  
        print('offset: ', offset)
        end = offset + BATCH_SIZE
        A_val = LDAMP.GenerateMeasurementMatrix(measurement_mode)

        batch_x_test = x_test[:, offset:end]
        batch_x_recon, batch_MSE_hist, batch_NMSE_hist, batch_PSNR_hist, batch_SSIM_hist = sess.run([x_hat, MSE_history, NMSE_history, PSNR_history, SSIM_history], feed_dict={x_true: batch_x_test, A_val_tf: A_val})
        recon.append(batch_x_recon)
        Final_PSNRs.append(batch_PSNR_hist[-1][0])
        Final_SSIMs.append(batch_SSIM_hist[-1][0])
        currentFinalPSNR.append(batch_PSNR_hist)
        currentFinalSSIM.append(batch_SSIM_hist)
    print('PSNR: ', Final_PSNRs)
    print('SSIM: ', Final_SSIMs)
    print('Mean PSNR: ', np.mean(Final_PSNRs))
    print('Mean SSIM: ', np.mean(Final_SSIMs))
    fig1 = plt.figure()
    plt.imshow(np.transpose(np.reshape(x_test[:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig2 = plt.figure()
    print('recon[:, 0]', recon[0][:, 0].shape)
    print('batch_x_recon[:, 0]', batch_x_recon[:, 0].shape)
    plt.imshow(np.transpose(np.reshape(recon[0][:, 0], (height_img, width_img))), interpolation='nearest', cmap='gray')
    plt.show()
    fig4 = plt.figure()
    plt.plot(range(n_DAMP_layers+1), np.mean(currentFinalPSNR[1],axis=1))
    plt.title("PSNR over DAMP layers")
    plt.show()
    print(range(n_DAMP_layers+1))
    print(currentFinalPSNR[2])