"""Training of models

This file define and execute two models based in pre-processed transcriptions and PHQ-8 results of persons.

Example:

        $ python3 ModelTraining.py

Notes:
    The training requires several resources like wordnet, glove word vectors and google word vectors, that are not included in this source code.

"""

import pickle
import sys
import numpy as np
import pandas as pd
import os
import ast
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("../")

print(os.path.dirname(__file__))

from datasets import DataSetGenerator
from model import GloVeModel
from model import Word2vecGoogleModel
from graphics import ModelGraphs
from datasets.DataReader import DataReader

data_path = os.getcwd().replace("app", "data/")


def confusion_matrix(model, x, y):
    prediction = model.predict(x, batch_size=None, verbose=0, steps=None)
    labels = ['none', 'mild', 'moderate', 'moderately severe', 'severe']

    max_prediction = np.argmax(prediction, axis=1)
    max_actual = np.argmax(y, axis=1)

    y_pred = pd.Categorical.from_codes(max_prediction, labels)
    y_actu = pd.Categorical.from_codes(max_actual, labels)

    return pd.crosstab(y_actu, y_pred, colnames=["Predicted"], rownames=["Actual"])


if __name__ == '__main__':
    """Training of models"""

    # reading the tokenized windows
    filename = data_path + "phrases_lp.csv"
    phrases_lp = pd.read_csv(filename, sep='\t')
    phrases_lp.columns = ['index', 'personId', 'answer', 't_answer']
    phrases_lp = phrases_lp.astype(
        {"index": float, "personId": float, "answer": np.ndarray, "t_answer": np.ndarray})

    # load of transcriptions and phq8 data
    data_reader = DataReader(data_path)
    data_reader.check_phq_target_distribution()

    ds_total = data_reader.ds_total
    ds_lp = pd.merge(data_reader.ds_total, phrases_lp, left_on='Participant_ID', right_on='personId')

    ds_lp_b = pd.merge(data_reader.ds_balanced, phrases_lp, left_on='Participant_ID', right_on='personId')

    with open(data_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    #glo_ve_model = GloVeModel(data_path)

    #glo_ve_history, glo_ve_model = glo_ve_model.train_model(tokenizer, ds_lp_b)

    google_model = Word2vecGoogleModel(data_path)
    google_history, google_model = google_model.train_model(tokenizer, ds_lp_b)

    #model_graphs = ModelGraphs()
    #model_graphs.plot_compare_accs(glo_ve_history,google_history, "GloVe Model", "Gloogle Model", "Accuracy history")

    #model_graphs.plot_compare_losses(glo_ve_history, google_history, "GloVe Model", "Gloogle Model", "Loss history")

    dataset_generator = DataSetGenerator(ds_lp_b)

    test = [ast.literal_eval(x) for x in dataset_generator.test['t_answer']]
    test_a = np.stack(test, axis=0)
    test_y = np.stack(dataset_generator.test['cat_level'], axis=0)

    df_confusion_google = confusion_matrix(google_model, test_a, test_y)

    score_google = google_model.evaluate(test_a, test_y, verbose=0)

    #glo_ve_model_json = glo_ve_model.to_json()

    #with open("glo_ve_model.json", "w") as json_file:
    #    json_file.write(glo_ve_model_json)

    #glo_ve_model.save_weights("glove_model_weights.h5")

    google_model_json = google_model.to_json()
    with open("google_model.json", "w") as json_file:
        json_file.write(google_model_json)

    google_model.save_weights("word2vec_model_weights.h5")

    print("Test google loss: {0: 2f}".format(score_google[0]))
    print("Test google accuracy: {0:.0%}".format(score_google[1]))

    sns.set()
    sns.heatmap(df_confusion_google, annot=True, fmt="#", cbar=False)
    plt.show()

    #df_confusion_glove = confusion_matrix(glo_ve_model, test_a, test_y)

    #score_glove = glo_ve_model.evaluate(test_a, test_y, verbose=0)

    #print("Test glove loss: {0: 2f}".format(score_glove[0]))
    #print("Test glove accuracy: {0:.0%}".format(score_glove[1]))

    #sns.set()
    #sns.heatmap(df_confusion_glove, annot=True, fmt="#", cbar=False)
    #plt.show()
