import os
import sys
import unittest

sys.path.append("..")

from util import predict_anxiety_level

data_path = os.getcwd().replace("tests", "data/")


class TestStringMethods(unittest.TestCase):
    def test_none(self):
        sen = "All is going right with the party, I'm happy to know new people"
        pred = predict_anxiety_level(data_path, sen, print_prediction=True, model_name="glove_model_balanced.h5")

        self.assertEqual(pred, 'none')

    def test_mild(self):
        sen = "I want ice cream and have some fries for lunch"
        pred = predict_anxiety_level(data_path, sen, print_prediction=True, model_name="glove_model_balanced.h5")

        self.assertEqual(pred, 'none')

    def test_severe(self):
        sen = "I'm afraid of losing my work, I don't have any money "
        pred = predict_anxiety_level(data_path, sen, print_prediction=True, model_name="glove_model_balanced.h5")

        self.assertEqual(pred, 'moderate')

    def test_moderate(self):
        sen = "I'm worried about my future, I'm afraid of it"
        pred = predict_anxiety_level(data_path, sen, print_prediction=True, model_name="glove_model_balanced.h5")

        self.assertEqual(pred, 'moderate')


if __name__ == '__main__':
    unittest.main()
