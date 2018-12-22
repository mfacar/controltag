import os
import sys
sys.path.append("..")

from util import predict_anxiety_level

data_path = os.getcwd().replace("app", "data/")

if __name__ == '__main__':
    sen = input('What are you thinking about?: ')
    predict_anxiety_level(data_path, sen, print_prediction=True)
