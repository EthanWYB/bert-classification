import tensorflow as tf
import pickle
import os
import numpy as np
from tensorflow import keras
from tokenization import FullTokenizer

data_dir_path = '../../Data/Customs_data/'
def get_data(tokenizer):
    with open(os.path.join(data_dir_path,'train_6_clean.tsv'), 'r', encoding='utf8') as f:
        train_data = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            line = line.split('\t')
            txt = line[1]
            input_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
            train_data.append(input_id)
        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                                     value=0,
                                                                     padding='post',
                                                                     maxlen=64)
    with open(os.path.join(data_dir_path,'dev_6_clean.tsv'), 'r', encoding='utf8') as f:
        dev_data = []
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            line = line.split('\t')
            txt = line[1]
            input_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
            dev_data.append(input_id)
        dev_data = keras.preprocessing.sequence.pad_sequences(dev_data,
                                                                    value=0,
                                                                    padding='post',
                                                                    maxlen=64)
    print(train_data)
    print(dev_data)
    return train_data, dev_data

def get_model(max_len=64, cls_num=2, embeddings=None):
        # input = keras.Input((max_len,768,))
        # input_1 = keras.layers.Dropout(0.4)(input)
        # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(128))(input_1)
        input = keras.Input((max_len,))
        embedding = keras.layers.Embedding(len(embeddings),
                                           768,
                                           weights=[embeddings],
                                           input_length=max_len,
                                           trainable=False
                                           )(input)

        # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(128))(embedding)

        c1 = keras.layers.Conv1D(128, 1, padding='same')(embedding)
        c2 = keras.layers.BatchNormalization()(c1)
        c2 = keras.layers.Activation(activation='relu')(c2)
        #
        c2 = keras.layers.Conv1D(128, 3, padding='same')(c2)
        #
        c3 = keras.layers.Conv1D(128, 3, padding='same')(embedding)
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


tokenizer = FullTokenizer(
    '/mnt/MountVolume2/nanjing-object/home/wangyanbo/Data/BertPretrainModel/chinese_L-12_H-768_A-12/vocab.txt',
    do_lower_case=True)
with open(os.path.join(data_dir_path,'embeddings.pkl'), 'rb') as f:
    embeddings = pickle.load(f)

train_data, dev_data = get_data(tokenizer)
train_label, dev_label, cls_num = get_label()

model = get_model(64, cls_num, embeddings)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    train_label,
                    epochs=50,
                    batch_size=128,
                    validation_data=(dev_data, dev_label))