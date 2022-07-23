import numpy as np
import pandas as pd
import glob
import os
import time

import tensorflow as tf
from tensorflow.keras import losses

from sklearn import manifold
import matplotlib.pyplot as plt

from network_module import dense_block, transition_block, Conv1DTranspose


class GPS_AE():
    def __init__(self):
        self.train_x, self.train_y, self.val_x, self.val_y = self.create_dataset()
        self.val_x2, self.val_y2 = self.create_night_dataset()

        self.val_x3 = np.concatenate((self.val_x, self.val_x2))
        self.val_y3 = np.concatenate((self.val_y, self.val_y2))

        BUFFER_SIZE = self.train_x.shape[0]
        BATCH_SIZE = self.train_x.shape[0]
        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
        self.train_dataset = self.train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        self.valid_dataset = tf.data.Dataset.from_tensor_slices((self.val_x3, self.val_y3)).batch(len(self.val_x3))

        self.model_down, self.model_encoder_decoder = self.create_model()
        # self.model_down.summary()
        # self.model_encoder_decoder.summary()
        
        self.model_loss = losses.MeanSquaredError()
        self.learning_rate_A = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
        self.optimizer_A = tf.optimizers.Adam() #SGD(learning_rate=learning_rate_A , momentum=1e-5)
        self.learning_rate_B = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
        self.optimizer_B = tf.optimizers.Adam(learning_rate=self.learning_rate_B)

        checkpoint_dir = './gps_checkpoints/checkpoints_0710_LPS_2_2'
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_0710_model_{epoch}")
        self.checkpoint = tf.train.Checkpoint(optimizerA=self.optimizer_A, model_encoder_decoder=self.model_encoder_decoder, model_down=self.model_down)

    def create_dataset(self):
        datas = glob.glob('transfer0508_2/*/*/*.txt')
        train_data = []
        train_label = [] 
        for data in datas:
            if data.find('timestamp') == -1:
                f = np.loadtxt(data,delimiter=" ").copy()
                f = f/[100,10,100,1,10,100,1e+13,1,1e+3,1e+15]
                f = f[[0,1,2,4]]
                train_data.append(f) #/100000000
                if data.split('/')[1] == "indoor":
                    train_label.append("indoor"+data.split('/')[-2])
                else:
                    train_label.append("outdoor"+data.split('/')[-2])
                

        train_data = np.array(train_data).astype('float32')
        train_label = np.array(pd.get_dummies(train_label)).astype('float32')

        train_val_split = np.random.rand(len(train_data)) < 0.70
        train_x = train_data[train_val_split]
        train_y = train_label[train_val_split]
        val_x = train_data[~train_val_split]
        val_y = train_label[~train_val_split]

        return train_x, train_y, val_x, val_y

    def create_night_dataset(self):
        datas = glob.glob('transfer0508_night_2/*/*/*.txt')
        train_data = []
        train_label = [] 
        for data in datas:
            if data.find('timestamp') == -1:
                f = np.loadtxt(data,delimiter=" ").copy()
                f = f/[100,10,100,1,10,100,1e+13,1,1e+3,1e+15]
                f = f[[0,1,2,4]]
                train_data.append(f)
                if data.split('/')[1] == "indoor":
                    train_label.append("indoor"+data.split('/')[-2])
                else:
                    train_label.append("outdoor"+data.split('/')[-2])
                
        train_data = np.array(train_data).astype('float32')
        train_label = np.array(pd.get_dummies(train_label)).astype('float32')

        train_val_split = np.random.rand(len(train_data)) < 0.70

        val_x2 = train_data[~train_val_split]
        val_y2 = train_label[~train_val_split]

        return val_x2, val_y2

    def create_model(self):
        inp = tf.keras.layers.Input(shape=(4), name='input_layer1')
        model = tf.keras.layers.Reshape((4,1))(inp)
        model = tf.keras.layers.Conv1D(32,3)(model)
        model = tf.keras.layers.AveragePooling1D(strides=1,padding="same")(model)
        model = dense_block(32,3,3,4)(model)
        model = transition_block(stride=1)(model)
        model = dense_block(4,3,3,4)(model)
        model_down = tf.keras.Model(inputs=[inp], outputs=model)

        inp = tf.keras.layers.Input(shape=(4), name='input_layer2')
        model = model_down(inp)
        model = Conv1DTranspose(32,3,1,'valid')(model)
        model = tf.keras.layers.AveragePooling1D(strides=1,padding='same')(model)
        model = Conv1DTranspose(1,3,1,'valid')(model)
        model = tf.keras.layers.AveragePooling1D(strides=1)(model)
        model = tf.keras.layers.Reshape((4,))(model)
        model_encoder_decoder = tf.keras.Model(inputs=[inp], outputs=model)

        return model_down, model_encoder_decoder

    def train_step(self, t_x, t_y):
        with tf.GradientTape() as AE_tape,tf.GradientTape() as ANN_tape:
            output = self.model_encoder_decoder(t_x, training=True)
            AE_loss = self.model_loss(output,t_x)

        gradients_AE = AE_tape.gradient(AE_loss, self.model_encoder_decoder.trainable_variables)
        self.optimizer_A.apply_gradients(zip(gradients_AE, self.model_encoder_decoder.trainable_variables))
        
        return np.array(AE_loss).mean()#,np.array(ANN_loss).mean()

    def validation(self, v_x, v_y):
        output = self.model_encoder_decoder(v_x)
        mse = losses.MeanSquaredError()
        AE_loss = mse(v_x, output)
        print("validation AE loss : {}\n".format(np.array(AE_loss).mean()))
        return np.array(AE_loss).mean()
        
    def train(self, epochs):
        minimum_delta = 0.00001
        history = {}
        history['val_loss'] = np.zeros(epochs)
        for epoch in range(epochs):
            start = time.time()
            all_AE = []
            for x, y in self.train_dataset:
                AE_loss = self.train_step(x, y)
                all_AE.append(AE_loss)
            print("train AE loss : {}".format(np.array(all_AE).mean()))
            loss = self.validation(self.val_x, self.val_y)
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
        val_latent = self.model_down(self.val_x3)
        val_latent = np.array(val_latent).reshape(val_latent.shape[0],-1)
        train_val_split = np.random.rand(len(val_latent)) < 0.5
        X_tsne = manifold.TSNE(n_components=2, init='pca', n_iter=500, method='exact').fit_transform(val_latent[train_val_split])
        y = self.val_y3[train_val_split].argmax(axis=1).reshape([-1,1])#.argmax(axis=1)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        plt.figure(figsize=(20, 20))
        cm = plt.cm.get_cmap('CMRmap')

        for i in range(X_norm.shape[0]):    
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i,0]), color=cm(y[i,0]*5), 
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        #plt.savefig("result/gps_pca_latent16_0710_LPS_2")
        plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    gps_AE = GPS_AE()
    gps_AE.train(20000)
    gps_AE.plot_result()