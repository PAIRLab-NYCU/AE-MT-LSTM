import numpy as np
import glob
import os
import time

import tensorflow as tf
from tensorflow.keras import  losses

from sklearn import manifold
import matplotlib.pyplot as plt

from network_module import dense_block, transition_block, Conv1DTranspose

class WiFi_AE():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        datas = glob.glob('0508_UIA/wifi_mag/CIPS-DataCollect/dataRssi_at_*.txt')

        self.train_dataset = self.create_train_dataset(datas)
        self.val_x, self.val_y, self.test_data, self.test_label = self.create_test_dataset(datas)

        self.model_down, self.model_encoder_decoder = self.create_model()
        # self.model_down.summary()

        self.model_loss = losses.MeanSquaredError()
        self.optimizer_A = tf.optimizers.Adam()

        checkpoint_dir = './wifi_checkpoints/checkpoints_0524_conv'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_dropout_model_{epoch}")
        self.checkpoint = tf.train.Checkpoint(optimizerA=self.optimizer_A, model_encoder_decoder=self.model_encoder_decoder, model_down=self.model_down)

    def create_train_dataset(self, datas):
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

    def create_test_dataset(self, datas):
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

    def create_model(self):
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

    def train_step(self, t_x, t_o, t_y):
        with tf.GradientTape() as AE_tape:
            output = self.model_encoder_decoder(t_x, training=True)
            AE_loss = self.model_loss(output,t_o)
        gradients_AE = AE_tape.gradient([AE_loss], self.model_encoder_decoder.trainable_variables)
        self.optimizer_A.apply_gradients(zip(gradients_AE, self.model_encoder_decoder.trainable_variables))

        return np.array(AE_loss).mean()

    def validation(self, v_x, v_o, v_y):
        output = self.model_encoder_decoder(v_x)
        AE_loss = self.model_loss(v_o, output)
        print("validation AE loss : {}\n".format(np.array(AE_loss).mean()))

        return np.array(AE_loss).mean()

    def train(self, epochs):
        minimum_delta = 0.00001
        history = {}
        history['val_loss'] = np.zeros(epochs)

        for epoch in range(epochs):
            start = time.time()
            all_AE = []
            for x,o,y in self.train_dataset:
                AE_loss = self.train_step(x,o,y)
                all_AE.append(AE_loss)
            print("train AE loss : {}".format(np.array(all_AE).mean()))
            loss = self.validation(self.val_x, self.val_x, self.val_y)
            history['val_loss'][epoch] = loss
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            if epoch > 200: #early stop
                    differences = np.abs(np.diff(history['val_loss'][epoch - 3:epoch], n = 1))
                    check =  differences > minimum_delta        
                    if np.all(check == False):
                        print(differences)
                        print("\n\nEarlyStopping Evoked! Stopping training\n\n")
                        break
            
            # print(f'Time for epoch {epoch + 1} is {time.time() - start:.4f} sec')

    def plot_result(self):
        val_latent = self.model_down(self.val_x)
        val_latent = np.array(val_latent).reshape(2500,-1)
        test = np.random.rand(len(val_latent)) < 1

        X_tsne = manifold.TSNE(n_components=2, init='pca', n_iter=5000, method='exact').fit_transform(val_latent[test])
        
        y = self.test_label[test].reshape([-1,1])
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        plt.figure(figsize=(10, 10))
        cm = plt.cm.get_cmap('CMRmap')
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i, 0]), color=cm(y[i, 0]*5), fontdict={'weight': 'bold', 'size': 9})

        plt.xticks([])
        plt.yticks([])
        plt.show()


if __name__ == "__main__":
    wifi_AE = WiFi_AE()
    wifi_AE.train(2000)
    wifi_AE.plot_result()