import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def SetNetworkParams(new_height_img, new_width_img,new_channel_img, new_filter_height,new_filter_width,\
                     new_num_filters,new_n_DnCNN_layers,new_n_DAMP_layers, new_sampling_rate,\
                     new_BATCH_SIZE,new_sigma_w,new_n,new_m,new_training,use_adaptive_weights=False):
    global height_img, width_img, channel_img, filter_height, filter_width, num_filters, n_DnCNN_layers, n_DAMP_layers,\
        sampling_rate, BATCH_SIZE, sigma_w, n, m, n_fp, m_fp, is_complex, training, adaptive_weights
    height_img = new_height_img
    width_img = new_width_img
    channel_img = new_channel_img
    filter_height = new_filter_height
    filter_width = new_filter_width
    num_filters = new_num_filters
    n_DnCNN_layers = new_n_DnCNN_layers
    n_DAMP_layers = new_n_DAMP_layers
    sampling_rate = new_sampling_rate
    BATCH_SIZE = new_BATCH_SIZE
    sigma_w = new_sigma_w
    n = new_n
    m = new_m
    n_fp = np.float32(n)
    m_fp = np.float32(m)
    is_complex=False
    adaptive_weights=use_adaptive_weights
    training=new_training


def ListNetworkParameters():
    print('height_img = ', height_img)
    print('width_img = ', width_img)
    print('channel_img = ', channel_img)
    print('filter_height = ', filter_height)
    print('filter_width = ', filter_width)
    print('num_filters = ', num_filters)
    print('n_DnCNN_layers = ', n_DnCNN_layers)
    print('n_DAMP_layers = ', n_DAMP_layers)
    print('sampling_rate = ', sampling_rate)
    print('BATCH_SIZE = ', BATCH_SIZE)
    print('sigma_w = ', sigma_w)
    print('n = ', n)
    print('m = ', m)

#Form the measurement operators
def GenerateMeasurementOperators(mode):
    global sparse_sampling_matrix
    global is_complex
    if mode=='gaussian':
        is_complex=False
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m, n))
        y_measured = tf.placeholder(tf.float32, [m, None])
        A_val_tf = tf.placeholder(tf.float32, [m, n])  

        def A_handle(A_vals_tf,x):
            return tf.matmul(A_vals_tf,x)

        def At_handle(A_vals_tf,z):
            return tf.matmul(A_vals_tf,z,adjoint_a=True)
    elif mode=='Fast-JL':
        is_complex=False
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)

        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.float32);
        inds = list(inds)
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])

        A_val_tf = tf.placeholder(tf.float32, [n, 1])
        def A_handle(A_val_tf, x):
            sign_vec = A_val_tf[0:n]
            signed_x = tf.multiply(sign_vec, x)
            signed_x = tf.reshape(signed_x, [height_img*width_img, BATCH_SIZE])
            signed_x=tf.transpose(signed_x)
            F_signed_x = mydct(signed_x, type=2, norm='ortho')
            F_signed_x=tf.transpose(F_signed_x)
            F_signed_x = tf.reshape(F_signed_x, [height_img * width_img, BATCH_SIZE])*np.sqrt(n_fp/m_fp)
            out = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,F_signed_x,adjoint_a=False)
            return out

        def At_handle(A_val_tf, z):
            sign_vec=A_val_tf[0:n]
            z_padded = tf.sparse_tensor_dense_matmul(sparse_sampling_matrix,z,adjoint_a=True)
            z_padded = tf.reshape(z_padded, [height_img*width_img, BATCH_SIZE])
            z_padded=tf.transpose(z_padded)
            Finv_z = myidct(z_padded,type=2,norm='ortho')
            Finv_z = tf.transpose(Finv_z)
            Finv_z = tf.reshape(Finv_z, [height_img*width_img, BATCH_SIZE])
            out = tf.multiply(sign_vec, Finv_z)*np.sqrt(n_fp/m_fp)
            return out
    else:
        raise ValueError('Measurement mode not recognized')
    return [A_handle, At_handle, A_val, A_val_tf]

def mydct(x,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE == 1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    y=tf.concat([x,tf.zeros([1,n],tf.float32)],axis=1)
    Y=tf.fft(tf.complex(y,tf.zeros([1,2*n],tf.float32)))
    Y=Y[:,:n]
    k = tf.complex(tf.range(n, dtype=tf.float32), tf.zeros(n, dtype=tf.float32))
    Y*=tf.exp(-1j*np.pi*k/(2.*n_fp))
    return tf.real(Y)/tf.sqrt(n_fp)*tf.sqrt(2.)

def myidct(X,type=2,norm='ortho'):
    assert type==2 and norm=='ortho', 'Currently only type-II orthonormalized DCTs are supported'
    assert BATCH_SIZE==1, 'Fast-JL measurement matrices currently only support batch sizes of one'
    temp0=tf.reverse(X,[-1])
    temp1=tf.manip.roll(temp0,shift=1,axis=1)
    temp2=temp1[:,1:]
    temp3=tf.pad(temp2,[[0,0],[1,0]],"CONSTANT")
    Z=tf.complex(X,-temp3)
    k = tf.complex(tf.range(n,dtype=tf.float32),tf.zeros(n,dtype=tf.float32))
    Z*=tf.exp(1j*np.pi*k/(2.*n_fp))
    temp4=tf.real(tf.ifft(Z))
    even_new=temp4[:,0:n//2]
    odd_new=tf.reverse(temp4[:,n//2:],[-1])
    x=tf.reshape(
        tf.transpose(tf.concat([even_new, odd_new], axis=0)),
        [1,n])
    return tf.real(x)*tf.sqrt(n_fp)*1/tf.sqrt(2.)

def GenerateMeasurementMatrix(mode):
    global sparse_sampling_matrix
    if mode == 'gaussian':
        A_val = np.float32(1. / np.sqrt(m_fp) * np.random.randn(m,n))
    elif mode == 'Fast-JL':
        A_val = np.zeros([n, 1])
        A_val[0:n] = np.sign(2*np.random.rand(n,1)-1)
        rand_col_inds=np.random.permutation(range(n))
        rand_col_inds=rand_col_inds[0:m]
        row_inds = range(m)
        inds=zip(row_inds,rand_col_inds)
        vals=tf.ones(m, dtype=tf.float32);
        inds = list(inds)
        sparse_sampling_matrix = tf.SparseTensor(indices=inds, values=vals, dense_shape=[m,n])
    else:
        raise ValueError('Measurement mode not recognized')
    return A_val

#Learned DAMP
def LDAMP(y,A_handle,At_handle,A_val,theta,x_true,tie,training=False,LayerbyLayer=True, test=False):
    z = y
    xhat = tf.zeros([n, BATCH_SIZE], dtype=tf.float32)
    MSE_history=[]
    NMSE_history=[]
    PSNR_history=[]
    SSIM_history = []
    (MSE_thisiter, NMSE_thisiter, PSNR_thisiter, SSIM_thisiter)=EvalError(xhat,x_true, test)
    MSE_history.append(MSE_thisiter)
    NMSE_history.append(NMSE_thisiter)
    PSNR_history.append(PSNR_thisiter)
    SSIM_history.append(SSIM_thisiter)
    for iter in range(n_DAMP_layers):
        r = xhat + At_handle(A_val,z)
        rvar = (1. / m_fp * tf.reduce_sum(tf.square(tf.abs(z)),axis=0))
        (xhat,dxdr)=DnCNN_outer_wrapper(r, rvar,theta,tie,iter,training=training,LayerbyLayer=LayerbyLayer)
        z = y - A_handle(A_val, xhat) + n_fp / m_fp * dxdr * z
        (MSE_thisiter, NMSE_thisiter, PSNR_thisiter, SSIM_thisiter) = EvalError(xhat, x_true,test)
        MSE_history.append(MSE_thisiter)
        NMSE_history.append(NMSE_thisiter)
        PSNR_history.append(PSNR_thisiter)
        SSIM_history.append(SSIM_thisiter)
    return xhat, MSE_history, NMSE_history, PSNR_history, r, rvar, dxdr, SSIM_history

def init_vars_DnCNN(init_mu,init_sigma):
    weights = [None] * n_DnCNN_layers
    biases = [None] * n_DnCNN_layers
    with tf.compat.v1.variable_scope("l0"):
        weights[0] = tf.Variable(
            tf.truncated_normal(shape=(filter_height, filter_width, channel_img, num_filters), mean=init_mu,
                                stddev=init_sigma), dtype=tf.float32, name="w")
    for l in range(1, n_DnCNN_layers - 1):
        with tf.compat.v1.variable_scope("l" + str(l)):
            weights[l] = tf.Variable(
                tf.truncated_normal(shape=(filter_height, filter_width, num_filters, num_filters), mean=init_mu,
                                    stddev=init_sigma), dtype=tf.float32, name="w")

    with tf.compat.v1.variable_scope("l" + str(n_DnCNN_layers - 1)):
        weights[n_DnCNN_layers - 1] = tf.Variable(
            tf.truncated_normal(shape=(filter_height, filter_width, num_filters, 1), mean=init_mu,
                                stddev=init_sigma), dtype=tf.float32,
            name="w")
    return weights, biases

## Evaluate Intermediate Error
def EvalError(x_hat,x_true,test):
    mse=tf.reduce_mean(tf.square(x_hat-x_true),axis=0)
    xnorm2=tf.reduce_mean(tf.square( x_true),axis=0)
    mse_thisiter=mse
    nmse_thisiter=mse/xnorm2
    psnr_thisiter=10.*tf.log(1./mse)/tf.log(10.)
    if test:
        x_hat_reshaped = tf.reshape(x_hat, [-1, 256, 256, 1])
        x_true_reshaped = tf.reshape(x_true, [-1, 256, 256, 1])
        ssim_thisiter = tf.image.ssim(x_hat_reshaped, x_true_reshaped, max_val=1.0)
    else:
        ssim_thisiter = None
    return mse_thisiter, nmse_thisiter, psnr_thisiter, ssim_thisiter

## Denoiser wrapper that selects which weights and biases to use
def DnCNN_outer_wrapper(r,rvar,theta,tie,iter,training=False,LayerbyLayer=True):
    with tf.compat.v1.variable_scope("Iter" + str(iter)):
        (xhat, dxdr) = DnCNN_wrapper(r, rvar, theta[iter], training=training,LayerbyLayer=LayerbyLayer)
    return (xhat, dxdr)

## Denoiser Wrapper that computes divergence
def DnCNN_wrapper(r,rvar,theta_thislayer,training=False,LayerbyLayer=True):
    xhat=DnCNN(r,rvar,theta_thislayer,training=training)
    r_abs = tf.abs(r, name=None)
    epsilon = tf.maximum(.001 * tf.reduce_max(r_abs, axis=0),.00001)
    eta=tf.random_normal(shape=r.get_shape(),dtype=tf.float32)
    r_perturbed = r + tf.multiply(eta, epsilon)
    xhat_perturbed=DnCNN(r_perturbed,rvar,theta_thislayer,training=training)
    eta_dx=tf.multiply(eta,xhat_perturbed-xhat)
    mean_eta_dx=tf.reduce_mean(eta_dx,axis=0)
    dxdrMC=tf.divide(mean_eta_dx,epsilon)
    return(xhat,dxdrMC)

def DnCNN(r,rvar, theta_thislayer,training=False):
    weights=theta_thislayer[0]

    r=tf.transpose(r)
    orig_Shape = tf.shape(r)
    shape4D = [-1, height_img, width_img, channel_img]
    r = tf.reshape(r, shape4D)
    layers = [None] * n_DnCNN_layers

    #############  First Layer ###############
    # Conv + Relu
    with tf.compat.v1.variable_scope("l0"):
        conv_out = tf.nn.conv2d(r, weights[0], strides=[1, 1, 1, 1], padding='SAME',data_format='NHWC') #NCHW works faster on nvidia hardware, however I only perform this type of conovlution once so performance difference will be negligible
        layers[0] = tf.nn.relu(conv_out)

    #############  2nd to 2nd to Last Layer ###############
    # Conv + BN + Relu
    for i in range(1,n_DnCNN_layers-1):
        with tf.compat.v1.variable_scope("l" + str(i)):
            conv_out  = tf.nn.conv2d(layers[i-1], weights[i], strides=[1, 1, 1, 1], padding='SAME') #+ biases[i]
            batch_out = tf.layers.batch_normalization(inputs=conv_out, training=training, name='BN', reuse=tf.AUTO_REUSE)
            layers[i] = tf.nn.relu(batch_out)

    #############  Last Layer ###############
    # Conv
    with tf.compat.v1.variable_scope("l" + str(n_DnCNN_layers - 1)):
        layers[n_DnCNN_layers-1]  = tf.nn.conv2d(layers[n_DnCNN_layers-2], weights[n_DnCNN_layers-1], strides=[1, 1, 1, 1], padding='SAME')

    x_hat = r-layers[n_DnCNN_layers-1]
    x_hat = tf.transpose(tf.reshape(x_hat,orig_Shape))
    return x_hat

## Create training data from images, with tf and function handles
def GenerateNoisyCSData_handles(x,A_handle,sigma_w,A_params):
    y = A_handle(A_params,x)
    y = AddNoise(y,sigma_w)
    return y

## Create training data from images, with tf
def AddNoise(clean,sigma):
    noise_vec=sigma*tf.random_normal(shape=clean.shape,dtype=tf.float32)
    noisy=clean+noise_vec
    noisy=tf.reshape(noisy,clean.shape)
    return noisy

##Create a string that generates filenames. Ensures consitency between functions
def GenLDAMPFilename(path,n_DAMP_layer_override=None,sampling_rate_override=None):
    if n_DAMP_layer_override:
        n_DAMP_layers_save=n_DAMP_layer_override
    else:
        n_DAMP_layers_save=n_DAMP_layers
    if sampling_rate_override:
        sampling_rate_save=sampling_rate_override
    else:
        sampling_rate_save=sampling_rate
    filename = path+"/SURE_DAMP_" + str(n_DnCNN_layers) + "DnCNNL_" + str(int(n_DAMP_layers_save)) + "DAMPL_Tie"+str(False)+"_LbyL"+str(True)+"_SR" +str(int(sampling_rate_save*100))
    return filename

## Count the total number of learnable parameters
def CountParameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('Total number of parameters: ')
    print(total_parameters)

## Calculate Monte Carlo SURE Loss
def MCSURE_loss(x_hat,div_overN,y,sigma_w):
    return tf.reduce_sum(tf.reduce_sum((y - x_hat) ** 2, axis=0) / n_fp -  sigma_w ** 2 + 2. * sigma_w ** 2 * div_overN)