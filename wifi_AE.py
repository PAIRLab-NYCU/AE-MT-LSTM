import numpy as np
import pandas as pd

import glob
import os
import time
import copy

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.layers import BatchNormalization, Activation, Conv1D, Concatenate, AveragePooling1D, Conv2DTranspose,Lambda

from sklearn import manifold, datasets
import matplotlib.pyplot as plt

noise = np.random.normal(scale = 0.05,size=(15))
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def create_train_dataset(datas):
    train_data = []
    origin_data = []
    train_label = []
    
    for data in datas:
        load_data = np.loadtxt(data)#f.read()
        load_data = load_data[:150,:15].astype(int)
        load_data[load_data==0] = -100
        load_data = load_data/100 + 1
        #print(load_data.shape)
        for i in range(len(load_data)):
            origin_data.append(load_data[i])
            mask = np.random.rand((15))
            mask = (mask>0.1).astype(int)
            noise = np.random.normal(scale = 0.05,size=(15))
            load_data[i]*mask
            load_data[i]+noise
            train_data.append(load_data[i])
            train_label.append(data.split("/")[-1].split("_")[-1].split(".")[0])

    origin_data = np.array(origin_data).astype('float32') 
    train_data = np.array(train_data).astype('float32')
    train_label = np.array(train_label).astype('int32')

    train_x = train_data
    train_o = origin_data
    train_y = tf.one_hot(train_label,50)
    BUFFER_SIZE = train_x.shape[0]
    BATCH_SIZE = int(train_x.shape[0]/2)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_x,train_o,train_y))
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    return train_dataset

def create_test_dataset(datas):
    test_data = []
    test_label = []
    for data in datas:
        load_data = np.loadtxt(data)#f.read()
        load_data = load_data[150:200,:15].astype(int)
        load_data[load_data==0] = -100
        load_data = load_data/100 + 1
        for i in range(len(load_data)):
            test_data.append(load_data[i])
            test_label.append(data.split("/")[-1].split("_")[-1].split(".")[0])
    test_data = np.array(test_data).astype('float32')
    test_label = np.array(test_label).astype('int32')
    val_x = test_data
    val_y = tf.one_hot(test_label,50)

    return val_x, val_y, test_data, test_label


def H_l(k, bottleneck_size, kernel_width):
    """ 
    A single convolutional "layer" as defined by Huang et al. Defined as H_l in the original paper
    
    :param k: int representing the "growth rate" of the DenseNet
    :param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    :param kernel_width: int representing the width of the main convolutional kernel
    :return a function wrapping the keras layers for H_l
    """

    use_bottleneck = bottleneck_size > 0
    num_bottleneck_output_filters = k * bottleneck_size

    def f(x):
        if use_bottleneck:
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
            x = Conv1D(
                num_bottleneck_output_filters,
                1,
                strides=1,
                padding="same",
                dilation_rate=1)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(
            k,
            kernel_width,
            strides=1,
            padding="same",
            dilation_rate=1)(x)
        return x
    return f

def dense_block(k, num_layers, kernel_width, bottleneck_size):
    """
    A single dense block of the DenseNet
    
    :param k: int representing the "growth rate" of the DenseNet
    :param num_layers: int represending the number of layers in the block
    :param kernel_width: int representing the width of the main convolutional kernel
    :param bottleneck_size: int representing the size of the bottleneck, as a multiple of k. Set to 0 for no bottleneck.
    :return a function wrapping the entire dense block
    """
    def f(x):
        layers_to_concat = [x]
        for _ in range(num_layers):
            x = H_l(k, bottleneck_size, kernel_width)(x)
            layers_to_concat.append(x)
            x = Concatenate(axis=-1)(copy.copy(layers_to_concat))
        return x
    return f

def transition_block(pool_size=2, stride=1, theta=0.5):
    """
    A single transition block of the DenseNet
    
    :param pool_size: int represending the width of the average pool
    :param stride: int represending the stride of the average pool
    :param theta: int representing the amount of compression in the 1x1 convolution. Set to 1 for no compression.
    :return a function wrapping the entire transition block
    """    
    assert theta > 0 and theta <= 1

    def f(x):
        num_transition_output_filters = int(int(x.shape[2]) * float(theta))
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv1D(
            num_transition_output_filters,
            1,
            strides=1,
            padding="same",
            dilation_rate=1)(x)
        x = AveragePooling1D(
            pool_size=pool_size,
            strides=stride)(x)
        return x
    return f

def Conv1DTranspose(filters, kernel_size, strides=2, padding='same'):
    def f(x):
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(x)
        x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
        return x
    return f

def create_model():
    inp = tf.keras.layers.Input(shape=(15), name='input_layer1')
    model = tf.keras.layers.Reshape((15,1))(inp)
    model = tf.keras.layers.Conv1D(32,3)(model)
    model = tf.keras.layers.AveragePooling1D(strides=1)(model)
    model = dense_block(32,3,3,4)(model)
    model = transition_block(stride=1)(model)
    model_down = tf.keras.Model(inputs=[inp], outputs=model)

    inp = tf.keras.layers.Input(shape=(15), name='input_layer2')
    model = model_down(inp)
    model = Conv1DTranspose(32,3,1,'valid')(model)
    model = tf.keras.layers.AveragePooling1D(strides=1,padding='same')(model)
    model = Conv1DTranspose(1,3,1,'valid')(model)
    model = tf.keras.layers.AveragePooling1D(strides=1,padding='same')(model)
    model = tf.keras.layers.Reshape((15,))(model)
    model_encoder_decoder = tf.keras.Model(inputs=[inp], outputs=model)

    return model_down, model_encoder_decoder

# @tf.function
def train_step(t_x,t_o,t_y):
    with tf.GradientTape() as AE_tape:
        output = model_encoder_decoder(t_x, training=True)
        AE_loss = model_loss(output,t_o)
    gradients_AE = AE_tape.gradient([AE_loss], model_encoder_decoder.trainable_variables)
    optimizer_A.apply_gradients(zip(gradients_AE, model_encoder_decoder.trainable_variables))

    return np.array(AE_loss).mean()

def validation(v_x,v_o,v_y):
    output = model_encoder_decoder(v_x)
    AE_loss = model_loss(v_o, output)
    print("AE loss : {}".format(np.array(AE_loss).mean()))

    return np.array(AE_loss).mean()

def train(epochs):
    minimum_delta = 0.00001
    history = {}
    history['val_loss'] = np.zeros(epochs)

    for epoch in range(epochs):
        start = time.time()
        all_AE = []
        for x,o,y in train_dataset:
            AE_loss = train_step(x,o,y)
            all_AE.append(AE_loss)
        print("train AE loss : {}".format(np.array(all_AE).mean()))
        loss = validation(val_x,val_x,val_y)
        history['val_loss'][epoch] = loss
        if epoch > 200: #early stop
                differences = np.abs(np.diff(history['val_loss'][epoch - 3:epoch], n = 1))
                check =  differences > minimum_delta        
                if np.all(check == False):
                    print(differences)
                    print("\n\nEarlyStopping Evoked! Stopping training\n\n")
                    break
        print(f'Time for epoch {epoch + 1} is {time.time() - start:.4f} sec')

def plot_result():
    val_latent = model_down(val_x)
    val_latent = np.array(val_latent).reshape(2500,-1)
    test = np.random.rand(len(val_latent)) < 1
    X_tsne = manifold.TSNE(n_components=2, init='pca', n_iter=5000, method='exact').fit_transform(val_latent[test])
    y = test_label[test].reshape([-1,1])
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10))
    cm = plt.cm.get_cmap('CMRmap')
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i,0]), color=cm(y[i,0]*5), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    datas = glob.glob('0508_UIA/wifi_mag/CIPS-DataCollect/dataRssi_at_*.txt')
    train_dataset = create_train_dataset(datas)
    val_x, val_y, test_data, test_label = create_test_dataset(datas)

    model_down, model_encoder_decoder = create_model()
    # model_down.summary()

    model_loss = losses.MeanSquaredError()
    optimizer_A = tf.optimizers.Adam()

    train(2000)
    
    plot_result()