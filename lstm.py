import numpy as np
import os
import datetime
import time

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses

from network_module import dense_block, transition_block, Conv1DTranspose

class DataPreprocess():
    def __init__(self):
        pass

    def get_dataset(self):
        wifi_train_data, mag_train_data, gps_train_data, tr_labelc, tr_label, tr_wgs_label = self.load_train_data()
        wifi_test_data, mag_test_data, gps_test_data, te_labelc, te_label, te_wgs_label = self.load_test_data()

        train_GPS, test_GPS = self.gps_preprocess(gps_train_data, gps_test_data)
        train_wifi, test_wifi = self.wifi_preprocess(wifi_train_data, wifi_test_data)

        train_dataset = self.create_train_dataset(train_GPS, train_wifi, mag_train_data, tr_labelc, tr_label, tr_wgs_label)
        valid_dataset = self.create_valid_dataset(test_GPS, test_wifi, mag_test_data, te_labelc, te_label, te_wgs_label)

        return train_dataset, valid_dataset

    def load_train_data(self):
        print("loading train data")
        wifi_train_data = np.loadtxt("data0601_LPS_2/train/w_train_data.txt").reshape(14001,21, 15)
        mag_train_data = np.loadtxt("data0601_LPS_2/train/m_train_data.txt").reshape(14001,21, 2)
        gps_train_data = np.loadtxt("data0601_LPS_2/train/g_train_data.txt").reshape(14001,21, 4)
        tr_labelc = np.loadtxt("data0601_LPS_2/train/train_labelc.txt").reshape(14001,21,2)
        tr_label = np.loadtxt("data0601_LPS_2/train/train_label.txt").reshape(14001,21)
        # train_iolabel = np.loadtxt("data/train/io_label.txt")
        tr_wgs_label = np.loadtxt("data0601_LPS_2/train/wgs_label.txt").reshape(14001,21,15)

        return wifi_train_data, mag_train_data, gps_train_data, tr_labelc, tr_label, tr_wgs_label

    def load_test_data(self):
        print("loading test data")
        wifi_test_data = np.loadtxt("data0601_LPS_2/test/w_test_data.txt").reshape(5999,21, 15)
        mag_test_data = np.loadtxt("data0601_LPS_2/test/m_test_data.txt").reshape(5999,21, 2)
        gps_test_data = np.loadtxt("data0601_LPS_2/test/g_test_data.txt").reshape(5999,21, 4)
        te_labelc = np.loadtxt("data0601_LPS_2/test/test_labelc.txt").reshape(5999,21,2)
        te_label = np.loadtxt("data0601_LPS_2/test/test_label.txt").reshape(5999,21)
        # te_iolabel = np.loadtxt("data/test/io_label.txt")
        te_wgs_label = np.loadtxt("data0601_LPS_2/test/wgs_label.txt").reshape(5999,21,15)

        return wifi_test_data, mag_test_data, gps_test_data, te_labelc, te_label, te_wgs_label

    def gps_preprocess(self, gps_train_data, gps_test_data):
        print("gps preprocess")

        inp = tf.keras.layers.Input(shape=(4), name='input_layer1')
        model = tf.keras.layers.Reshape((4,1))(inp)
        model = tf.keras.layers.Conv1D(32,3)(model)
        model = tf.keras.layers.AveragePooling1D(strides=1,padding="same")(model)
        model = dense_block(32,3,3,4)(model)
        model = transition_block(stride=1)(model)
        model = dense_block(32,3,3,4)(model)
        model_down_gps = tf.keras.Model(inputs=[inp], outputs=model)
        # load gps trained model
        path = "gps_checkpoints/checkpoints_0601_LPS_2/"
        latest = tf.train.latest_checkpoint(path)

        checkpoint = tf.train.Checkpoint(model_down=model_down_gps)
        checkpoint.restore(latest).expect_partial()

        train_GPS = []
        test_GPS = []

        for i in range(14001):
            GPS = model_down_gps(gps_train_data[i].astype('float32'))
            train_GPS.append(np.array(GPS).reshape(GPS.shape[0],-1))

        for i in range(5999):
            GPS = model_down_gps(gps_test_data[i].astype('float32'))
            test_GPS.append(np.array(GPS).reshape(GPS.shape[0],-1))

        print("gps shape : ", train_GPS[0].shape)
        return train_GPS, test_GPS

    def wifi_preprocess(self, wifi_train_data, wifi_test_data):
        print("wifi preprocess")

        inp = tf.keras.layers.Input(shape=(15), name='input_layer1')
        model = tf.keras.layers.Reshape((15,1))(inp)
        model = tf.keras.layers.Conv1D(32,3)(model)
        model = tf.keras.layers.AveragePooling1D(strides=1)(model)
        model = dense_block(32,3,3,4)(model)
        model = transition_block(stride=1)(model)
        model_down_wifi = tf.keras.Model(inputs=[inp], outputs=model)

        path = "wifi_checkpoints/checkpoints_0524_conv/"
        latest = tf.train.latest_checkpoint(path)

        checkpoint = tf.train.Checkpoint(model_down=model_down_wifi)
        checkpoint.restore(latest).expect_partial()

        train_wifi = []
        test_wifi = []

        for i in range(14001):
            wifi = model_down_wifi(wifi_train_data[i].astype('float32'))
            train_wifi.append(np.array(wifi).reshape(wifi.shape[0],-1))

        for i in range(5999):
            wifi = model_down_wifi(wifi_test_data[i].astype('float32'))
            test_wifi.append(np.array(wifi).reshape(wifi.shape[0],-1))

        print("wifi shape : ", train_wifi[0].shape)
        return train_wifi, test_wifi

    def create_train_dataset(self, train_GPS, train_wifi, mag_train_data, tr_labelc, tr_label, tr_wgs_label):
        print("train dataset creating")

        train_data = []
        train_label = []
        train_iolabel = []
        train_wgs_label = []

        for i in range(14001):
            wifi = train_wifi[i]
            g_mag = mag_train_data[i]
            GPS = train_GPS[i]
            lstm_input = np.concatenate([wifi,g_mag,GPS],1)
            for j in range (17):
                train_data.append(lstm_input[j:j+5])
                train_label.append(tr_labelc[i][j+4])
                if tr_label[i][j+4] < 26:
                    train_iolabel.append(0)
                else:
                    train_iolabel.append(1)
                train_wgs_label.append(np.array(tr_wgs_label[i][j:j+5]).reshape(-1))

        train_data = np.array(train_data).astype('float32')
        train_label = np.array(train_label).astype('float32')
        io_label = np.array(train_iolabel).astype('int32')
        wgs_label = np.array(train_wgs_label).astype('int32')
        io_label = tf.one_hot(io_label, 2)

        print(train_data.shape, train_label.shape, io_label.shape, wgs_label.shape)

        BUFFER_SIZE = train_data.shape[0]
        BATCH_SIZE = int(train_data.shape[0]//100) + 1
        
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label, io_label, wgs_label))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        return train_dataset

    def create_valid_dataset(self, test_GPS, test_wifi, mag_test_data, te_labelc, te_label, te_wgs_label):
        print("test dataset creating")

        test_data = []
        test_label = []
        test_iolabel = []
        test_wgs_label = []

        for i in range(5999):
            wifi = test_wifi[i]
            g_mag = mag_test_data[i]
            GPS = test_GPS[i] #np.array(GPS).reshape(GPS.shape[0],-1)
            lstm_input = np.concatenate([wifi,g_mag,GPS],1)
            for j in range (17):
                test_data.append(lstm_input[j:j+5])
                test_label.append(te_labelc[i][j+4])
                if te_label[i][j+4] < 26:
                    test_iolabel.append(0)
                else:
                    test_iolabel.append(1)
                test_wgs_label.append(np.array(te_wgs_label[i][j:j+5]).reshape(-1))

        test_data = np.array(test_data).astype('float32')
        test_label = np.array(test_label).astype('float32')
        test_io_label = np.array(test_iolabel).astype('int32')
        test_wgs_label = np.array(test_wgs_label).astype('int32')
        test_io_label = tf.one_hot(test_io_label,2)

        valid_dataset = tf.data.Dataset.from_tensor_slices((test_data,test_label,test_io_label,test_wgs_label)).batch(len(test_data))

        return valid_dataset


class LSTM_Model():
    def __init__(self):
        data_preprocess = DataPreprocess()
        self.train_dataset, self.test_dataset = data_preprocess.get_dataset()

        self.optimizer_A = tf.optimizers.Nadam()
        self.model = self.create_model()

    def euclidean_distance_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def model_loss(self, output, t_y, t_io, t_wgs):
        output_categorial , output_io, output_wgs = output
        coor_loss = self.euclidean_distance_loss(t_y,output_categorial)
        io_loss = losses.binary_crossentropy(t_io,output_io)
        wgs_loss = losses.binary_crossentropy(t_wgs,output_wgs)

        return coor_loss, io_loss, wgs_loss

    def create_model(self):
        lstm_input = tf.keras.layers.Input(shape=(5,866), name='input_layer_lstm')
        lstm1 = tf.keras.layers.LSTM(20,return_sequences=True)(lstm_input)
        lstm2 = tf.keras.layers.LSTM(20,return_sequences=False)(lstm1)
        fc3 = tf.keras.layers.Dense(2,name='fc3')(lstm2)
        io3 = tf.keras.layers.Dense(2,activation='sigmoid',name='io3')(lstm2)
        wgs3 = tf.keras.layers.Dense(75,activation='sigmoid',name='wgs3')(lstm2)
        model_lstm = tf.keras.Model(inputs=[lstm_input], outputs=[fc3,io3,wgs3],name = "lstm")

        return model_lstm

    def train_step(self, t_x, t_y, t_io, t_wgs):
        lstm_vars = self.model.trainable_variables
    
        with tf.GradientTape() as lstm_tape, tf.GradientTape() as io_tape, tf.GradientTape() as wgs_tape:
            output = self.model(t_x, training=True)
            coor_loss, io_loss, wgs_loss = self.model_loss(output,t_y,t_io,t_wgs)
            coor_loss = tf.multiply(0.5,coor_loss)

        # rand = np.random.rand()
        # if rand < 0.4:
        gradients = lstm_tape.gradient([coor_loss,io_loss,wgs_loss], lstm_vars)# + categorial_vars)
        self.optimizer_A.apply_gradients(zip(gradients, lstm_vars))# + categorial_vars))

        return np.array(coor_loss).mean(),np.array(io_loss).mean(),np.array(wgs_loss).mean()

    def validation(self, v_x, v_y, v_io, v_wgs):
        output = self.model(v_x)
        output_label,output_io, output_wgs = output
        coor_loss = self.euclidean_distance_loss(v_y,output_label)
        io_loss = losses.binary_crossentropy(v_io,output_io)
        wgs_loss = losses.binary_crossentropy(v_wgs,output_wgs)

        t = (np.sum(np.square(output_label - v_y),axis=1))
        e3 = 0
        e2 = 0
        e1 = 0
        for tt in t :
            if tt <= 3:
                e3 += 1
            if tt <= 2:
                e2 += 1
            if tt <= 1 :
                e1 += 1
        print("coor loss : {}, io loss : {}, wgs loss : {}, distance error : {}, e3:{}, e2:{}, e1:{}\n".format(np.array(coor_loss).mean(),np.array(io_loss).mean(),np.array(wgs_loss).mean(),(np.sqrt(np.sum(np.square(output_label - v_y),axis=1))).mean(),e3/v_y.shape[0],e2/v_y.shape[0],e1/v_y.shape[0]))
        
        return np.array(coor_loss).mean(), np.array(io_loss).mean(),np.array(wgs_loss).mean()

    def train(self, epochs):
        checkpoint_dir = './lstm_checkpoints/AE_LSTM_checkpoints_0601_coor_LPS_2'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_dropout_model_{epoch}")
        checkpoint = tf.train.Checkpoint(optimizerA=self.optimizer_A, model_lstm=self.model)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = '/root/notebooks/tensorflow/logs/gradient_tape/' + current_time + '/train'
        test_log_dir = '/root/notebooks/tensorflow/logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        minimum_delta = 0.0001
        history = {}
        history['val_loss'] = np.zeros(epochs)

        for epoch in range(epochs):
            print("epoch : ", epoch)
            start = time.time()
            all_coor = []
            all_io = []
            all_wgs = []

            for x,y,io,wgs in self.train_dataset:
                # print(x.shape)
                coor_loss,io_loss,wgs_loss = self.train_step(x,y,io,wgs)
                all_coor.append(coor_loss)
                all_io.append(io_loss)
                all_wgs.append(wgs_loss)
            
            print("train categorial loss : {}, train io loss : {}, train wgs loss : {}".format(np.array(all_coor).mean(),np.array(all_io).mean(),np.array(all_wgs).mean()))

            with train_summary_writer.as_default():
                tf.summary.scalar('coor loss', np.array(all_coor).mean(), step=epoch)
                tf.summary.scalar('io loss', np.array(all_io).mean(), step=epoch)
                tf.summary.scalar('wgs loss', np.array(all_wgs).mean(), step=epoch)

            for (val_x, val_y, val_io, val_wgs) in self.test_dataset:
                loss, io_loss, wgs_loss = self.validation(val_x, val_y, val_io, val_wgs)
                # print("val end")
            
            with test_summary_writer.as_default():
                tf.summary.scalar('coor loss', loss, step=epoch)
                tf.summary.scalar('io loss', io_loss, step=epoch)
                tf.summary.scalar('wgs loss', wgs_loss, step=epoch)

            # print("sum end")
            history['val_loss'][epoch] = loss
            checkpoint.save(file_prefix=checkpoint_prefix)

            if epoch > 200: #early stop
                differences = np.abs(np.diff(history['val_loss'][epoch - 5:epoch], n = 1))
                check =  differences > minimum_delta 

                if np.all(check == False):
                    print(differences)
                    print("\n\nEarlyStopping Evoked! Stopping training\n\n")
                    break

            # print("learning rate A : ",optimizer_A._decayed_lr(tf.float32))
            # print("learning rate B : ",optimizer_B._decayed_lr(tf.float32))
            # print(f'Time for epoch {epoch + 1} is {time.time() - start:.4f} sec')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    lstm_model = LSTM_Model()
    lstm_model.train(3000)
