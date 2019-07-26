import tensorflow as tf
import pickle
import numpy as np
import os
from tensorflow import keras

data_dir_path = '../../Data/Customs_data/'
def get_data(layer=0):
    with open(os.path.join(data_dir_path,'train_fea.pkl'), 'rb') as f:
        data_sample = pickle.load(f)
        train_data = np.zeros((len(data_sample), 64, 768))
        for (i, sample) in enumerate(data_sample):
            # print(len(data_sample[i]))
            for j in range(len(data_sample[i])):
                train_data[i, j, :] = data_sample[i][j][layer]
    with open(os.path.join(data_dir_path,'dev_fea.pkl'), 'rb') as f:
        data_sample = pickle.load(f)
        dev_data = np.zeros((len(data_sample), 64, 768))
        for (i,sample) in enumerate(data_sample):
            for j in range(len(data_sample[i])):
                dev_data[i,j,:] = data_sample[i][j][layer]
    return train_data, dev_data

def get_model(max_len=64, cls_num=2):
        input = keras.Input((max_len,768,))
        # input_1 = keras.layers.Dropout(0.4)(input)
        # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(128))(input_1)
        # x = keras.layers.Dropout(0.4)(x)

        c1 = keras.layers.Conv1D(128, 1, padding='same')(input)
        c2 = keras.layers.BatchNormalization()(c1)
        c2 = keras.layers.Activation(activation='relu')(c2)
        #
        c2 = keras.layers.Conv1D(128, 3, padding='same')(c2)
        #
        c3 = keras.layers.Conv1D(128, 3, padding='same')(input)
        c4 = keras.layers.BatchNormalization()(c3)
        c4 = keras.layers.Activation(activation='relu')(c4)
        #
        c4 = keras.layers.Conv1D(128, 5, padding='same')(c4)

        mid = keras.layers.Concatenate()([c1, c2, c3, c4])

        mid = keras.layers.BatchNormalization()(mid)
        mid = keras.layers.Dropout(0.5)(mid)
        mid = keras.layers.Activation(activation='relu')(mid)

        mid = keras.layers.Flatten()(mid)
        x = keras.layers.Dropout(0.4)(mid)


        output = keras.layers.Dense(cls_num,
                                    activation='softmax')(x)
        # output = keras.layers.Dropout(0.4)(output)
        model = keras.Model(inputs=input,
                            outputs=output)
        model.summary()
        return model

def get_label():
    label_dict = {}
    train_label = []
    dev_label = []
    with open(os.path.join(data_dir_path,'train_6_clean.tsv'), 'r', encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            line = line.split('\t')
            label = line[0]
            if label not in label_dict.keys():
                label_dict[label] = len(label_dict)
            train_label.append(label_dict[label])
    with open(os.path.join(data_dir_path,'dev_6_clean.tsv'), 'r', encoding='utf8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            line = line.split('\t')
            label = line[0]
            if label not in label_dict.keys():
                label_dict[label] = len(label_dict)
            dev_label.append(label_dict[label])
    return train_label, dev_label, len(label_dict)

train_data, dev_data = get_data()
train_label, dev_label, cls_num = get_label()

model = get_model(64, cls_num)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    train_label,
                    epochs=50,
                    batch_size=128,
                    validation_data=(dev_data, dev_label))