"""Control TAG execution

This console application make use of the training steps described in the readme file

Example:

        $ python3 App.py glove_model.h5


Attributes:
    model name (str): Name of the model that will be used to evaluate the entered phrase.

Todo:
    * Reproduction of music in accordance to the anxiety level.
"""

import os
import sys
import getopt

sys.path.append("..")

from util import predict_anxiety_level

data_path = os.getcwd().replace("app", "data/")


def main(argv):
    """Evaluation of specified model with a text of input"""
    opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    model = args[0] if len(args) > 0 else "glove"

    model_name = "glo_ve_model.json" if model == "glove" else "google_model.json"
    weights_name = "glove_model_weights.h5" if model == "glove" else "word2vec_model_weights.h5"

    sen = input('What are you thinking about?: ')
    predict_anxiety_level(data_path, sen, print_prediction=True, model_name=model_name, weights_name=weights_name)


if __name__ == '__main__':
    main(sys.argv[1:])
