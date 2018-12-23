import itertools
import pickle
import numpy as np

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from util.TextPreProcessor import TextPreProcessor

labels = ['none', 'mild', 'moderate', 'moderately severe', 'severe']


def make_input_from_text(text, tokenizer, windows_size):
    text_pre_processor = TextPreProcessor()
    word_list = text_pre_processor.text_to_wordlist(text)
    sequences = tokenizer.texts_to_sequences([word_list])
    sequences_input = list(itertools.chain(*sequences))
    sequences_input = pad_sequences([sequences_input], value=0, padding="post", maxlen=windows_size).tolist()
    input_a = np.asarray(sequences_input)
    return input_a


def predict_anxiety_level(data_path, text, print_prediction, model_name="glove_model_balanced.h5"):
    model_path = data_path + model_name
    model = load_model(model_path)
    with open(data_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    windows_size = 10
    input_a = make_input_from_text(text, tokenizer, windows_size)

    print("*" * 50)
    print("Phrase: {}".format(text))
    return test_model(model, input_a, print_prediction)


def test_model(model, input_a, print_prediction):
    pred = model.predict(input_a, batch_size=None, verbose=0, steps=None)

    if print_prediction:
        print("Predictions: {}".format(pred))
    predicted_class = np.argmax(pred[0])
    print("Anxiety level: {}".format(labels[predicted_class]))
    return labels[predicted_class]
