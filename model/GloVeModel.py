import numpy as np
import pandas as pd
import ast

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras_preprocessing.text import Tokenizer

from app.GlobalConstants import WINDOWS_SIZE
from datasets import DataSetGenerator


class GloVeModel:
    FILE_NAME = 'glove.6B.100d.txt'

    def __init__(self, path):
        path = path + self.FILE_NAME
        self.embeddings_index = dict()
        print(path)
        f = open(path, encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()

    def fill_embedding_matrix(self, vocab_size, tokenizer: Tokenizer):
        embedding_matrix = np.zeros((vocab_size + 1, 100))
        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def train_model(self, tokenizer: Tokenizer, ds_total: pd.DataFrame):
        vocab_size = len(tokenizer.word_index)
        windows_size = WINDOWS_SIZE
        embedding_matrix_lp = self.fill_embedding_matrix(vocab_size, tokenizer)

        dataset_generator = DataSetGenerator(ds_total)

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        train = [ast.literal_eval(x) for x in dataset_generator.train['t_answer']]
        train_a = np.stack(train, axis=0)
        dev = [ast.literal_eval(x) for x in dataset_generator.dev['t_answer']]
        dev_a = np.stack(dev, axis=0)
        train_y = np.stack(dataset_generator.train['cat_level'], axis=0)
        dev_y = np.stack(dataset_generator.dev['cat_level'], axis=0)

        answer_inp = Input(shape=(windows_size, ))
        embedding_size_glove = 100
        answer_emb1 = Embedding(vocab_size+1, embedding_size_glove, weights=[embedding_matrix_lp],
                                input_length=windows_size, trainable=False)(answer_inp)

        bt = BatchNormalization()(answer_emb1)
        lstm = LSTM(embedding_size_glove, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(bt)

        dense1 = Dense(units=256, activation="relu")(lstm)
        dense2 = Dense(units=256, activation="relu")(dense1)

        flatten = Flatten()(dense2)

        out = Dense(5,  activation='softmax')(flatten)

        model = Model(inputs=[answer_inp], outputs=[out])
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint('glove_model_balanced.h5', verbose=1, monitor='val_loss', save_best_only=True,
                                     mode='auto')

        model_glove_hist = model.fit([train_a], train_y,
                                     validation_data=([dev_a], dev_y),
                                     epochs=100, batch_size=64, shuffle=True, verbose=True,
                                     callbacks=[early_stopping, checkpoint])

        return model_glove_hist, model

    def train_two_losses_model(self, tokenizer: Tokenizer, ds_total: pd.DataFrame):
        vocab_size = len(tokenizer.word_index)
        windows_size = WINDOWS_SIZE

        embedding_matrix_lp = self.fill_embedding_matrix(vocab_size, tokenizer)

        dataset_generator = DataSetGenerator(ds_total)

        train = [ast.literal_eval(x) for x in dataset_generator.train['t_answer']]
        train_a = np.stack(train, axis=0)
        dev = [ast.literal_eval(x) for x in dataset_generator.dev['t_answer']]
        dev_a = np.stack(dev, axis=0)
        train_y = np.stack(dataset_generator.train['cat_level'], axis=0)
        dev_y = np.stack(dataset_generator.dev['cat_level'], axis=0)

        train_phq_score = dataset_generator.train['PHQ8_Score']
        dev_phq_score = dataset_generator.dev['PHQ8_Score']
        answer_inp = Input(shape=(windows_size, ))
        embedding_size_glove = 100
        answer_emb1 = Embedding(vocab_size + 1, embedding_size_glove, weights=[embedding_matrix_lp],
                                input_length=windows_size, trainable=False)(answer_inp)

        bt = BatchNormalization()(answer_emb1)
        lstm = LSTM(embedding_size_glove, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(bt)

        dense1 = Dense(units=256, activation="relu")(lstm)
        dense2 = Dense(units=256, activation="relu")(dense1)

        flatten = Flatten()(dense2)

        preds_1 = Dense(5, activation='softmax')(flatten)
        preds_2 = Dense(1, activation='sigmoid')(flatten)

        model = Model(inputs=[answer_inp],
                      outputs=[preds_1, preds_2])

        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'],
                      optimizer='nadam', loss_weights=[1., 0.2],
                      metrics=['acc'])

        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        model_checkpoint = ModelCheckpoint('glove_two_losses_model.h5', verbose=1, monitor='val_loss', save_best_only=True,
                                     mode='auto')

        two_losses_hist = model.fit([train_a], [train_y, train_phq_score],
                         validation_data=([dev_a], [dev_y, dev_phq_score]),
                         epochs=100, batch_size=64, shuffle=True, callbacks=[early_stopping, model_checkpoint])

        return two_losses_hist, model
