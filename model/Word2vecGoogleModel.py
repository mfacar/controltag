import numpy as np
import gensim


class Word2vecGoogleModel:
    FILE_NAME = 'GoogleNews-vectors-negative300.bin'

    def __init__(self, path='/Users/mercyfalconi/PycharmProjects/ControlTAC/data/'):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(
            '/content/drive/My Drive/Colab Notebooks/GoogleNews-vectors-negative300.bin', binary=True)

        self.embeddings_index = dict()
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
