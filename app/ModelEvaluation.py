"""Evaluation of model

This file evaluate a model against test data

Example:

        $ python3 ModelEvaluation.py

Notes:
    The model are recovered from previous training

"""

import ast
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from keras.engine.saving import load_model

sys.path.append("../")

print(os.path.dirname(__file__))

from datasets import DataSetGenerator

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

    dataset_generator = DataSetGenerator(ds_lp_b)

    test = [ast.literal_eval(x) for x in dataset_generator.test['t_answer']]
    test_a = np.stack(test, axis=0)
    test_y = np.stack(dataset_generator.test['cat_level'], axis=0)

    model_path = "glove_model_balanced.h5"
    model = load_model(model_path)

    df_confusion = confusion_matrix(model, test_a, test_y)

    score = model.evaluate(test_a, test_y, verbose=0)

    print("Test loss: {0: 2f}".format(score[0]))
    print("Test accuracy: {0:.0%}".format(score[1]))

    sns.set()
    sns.heatmap(df_confusion, annot=True, fmt="#", cbar=False)

    df_confusion
