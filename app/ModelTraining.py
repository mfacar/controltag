import pickle
import sys
import numpy as np
import pandas as pd
import os

sys.path.append("../")

print(os.path.dirname(__file__))

from datasets import DataSetGenerator
from model import GloVeModel
from model import Word2vecGoogleModel
from graphics import ModelGraphs
from datasets.DataReader import DataReader

data_path = os.getcwd().replace("app", "data/")

if __name__ == '__main__':
    # read transcripts and load in dataframe
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

    with open(data_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    glo_ve_model = GloVeModel(data_path)

    glo_ve_history, glo_ve_model = glo_ve_model.train_model(tokenizer, ds_lp_b)

    google_model = Word2vecGoogleModel(data_path)
    google_history, google_model = google_model.train_model(tokenizer, ds_lp_b)

    model_graphs = ModelGraphs()
    model_graphs.plot_compare_accs(glo_ve_history,google_history, "GloVe Model", "Gloogle Model", "Accuracy history")

    model_graphs.plot_compare_losses(glo_ve_history, google_history, "GloVe Model", "Gloogle Model", "Loss history")
