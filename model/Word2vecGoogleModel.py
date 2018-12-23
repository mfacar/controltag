import numpy as np
import pandas as pd
import ast
import gensim
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Input, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras_preprocessing.text import Tokenizer

from datasets import DataSetGenerator


class Word2vecGoogleModel:
    FILE_NAME = 'GoogleNews-vectors-negative300.bin'
    EMBEDDING_DIM = 300

    def __init__(self, path):
        path = path + self.FILE_NAME
        print(path)
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

    def fill_embedding_matrix_google(self, vocab_size, tokenizer):
        embedding_matrix = np.zeros((vocab_size+1, 300))
        for word, i in tokenizer.word_index.items():
            if word in self.word2vec.vocab:
                embedding_matrix[i] = self.word2vec.word_vec(word)
        return embedding_matrix

    def train_model(self, tokenizer: Tokenizer, ds_total: pd.DataFrame):
        windows_size = 10
        vocab_size = len(tokenizer.word_index)

        dataset_generator = DataSetGenerator(ds_total)
        embedding_matrix_gg = self.fill_embedding_matrix_google(vocab_size, tokenizer)

        train = [ast.literal_eval(x) for x in dataset_generator.train['t_answer']]
        train_a = np.stack(train, axis=0)
        dev = [ast.literal_eval(x) for x in dataset_generator.dev['t_answer']]
        dev_a = np.stack(dev, axis=0)
        train_y = np.stack(dataset_generator.train['cat_level'], axis=0)
        dev_y = np.stack(dataset_generator.dev['cat_level'], axis=0)

        num_lstm = np.random.randint(175, 275)
        num_dense = np.random.randint(100, 150)
        rate_drop_lstm = 0.15 + np.random.rand() * 0.25
        rate_drop_dense = 0.15 + np.random.rand() * 0.25

        act = 'relu'

        stamp = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)

        embedding_layer = Embedding(vocab_size + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[embedding_matrix_gg],
                                    input_length=windows_size,
                                    trainable=False)

        lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

        answer_inp = Input(shape=(windows_size,), dtype='int32')
        embedded_sequences_1 = embedding_layer(answer_inp)
        x1 = lstm_layer(embedded_sequences_1)

        merged = x1
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(num_dense, activation=act)(merged)
        merged = Dropout(rate_drop_dense)(merged)
        merged = BatchNormalization()(merged)

        out = Dense(5, activation='softmax')(merged)

        model_gg_1 = Model(inputs=[answer_inp], outputs=[out])
        model_gg_1.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        model_gg_1.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        bst_model_path = stamp + '.google.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

        model_google_hist = model_gg_1.fit([train_a], train_y,
                                           validation_data=([dev_a], dev_y),
                                           epochs=50, batch_size=64, shuffle=True,
                                           callbacks=[early_stopping, model_checkpoint])

        return model_google_hist, model_gg_1
