import sys
import os
import numpy as np
import pickle

from datasets import DataSetGenerator
from model import GloVeModel
from graphics import ModelGraphs

sys.path.append("..")

import pandas as pd

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

    dataset_generator = DataSetGenerator(ds_lp)

    with open(data_path + 'tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    glo_ve_model = GloVeModel(data_path)
    glo_ve_history, glo_ve_model = glo_ve_model.train_model(tokenizer, ds_lp)

    model_graphs = ModelGraphs()
    model_graphs.plot_acc(glo_ve_history, "GloVe Model")

