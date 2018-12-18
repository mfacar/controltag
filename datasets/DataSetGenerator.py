import pandas as pd
from datasets import DataReader


class DataSetGenerator:
    ds_total: pd.DataFrame

    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame

    def __init__(self, ds_total):
        self.train, self.dev, self.test = self.__distribute_instances__(ds_total)

    @staticmethod
    def __distribute_instances__(ds):
        ds_shuffled = ds.sample(frac=1)
        none_ds, mild_ds, moderate_ds, moderate_severe_ds, severe_ds = DataReader.split_by_phq_level(ds_shuffled)
        split = [70, 14, 16]
        eq_ds = {}
        prev_none = prev_mild = prev_moderate = prev_moderate_severe = prev_severe = 0

        for p in split:
            last_none = min(len(none_ds), prev_none + round(len(none_ds) * p / 100))
            last_mild = min(len(mild_ds), prev_mild + round(len(mild_ds) * p / 100))
            last_moderate = min(len(moderate_ds), prev_moderate + round(len(moderate_ds) * p / 100))
            last_moderate_severe = min(len(moderate_severe_ds),
                                       prev_moderate_severe + round(len(moderate_severe_ds) * p / 100))
            last_severe = min(len(severe_ds), prev_severe + round(len(severe_ds) * p / 100))
            eq_ds["d" + str(p)] = pd.concat([none_ds[prev_none: last_none], mild_ds[prev_mild: last_mild],
                                             moderate_ds[prev_moderate: last_moderate],
                                             moderate_severe_ds[prev_moderate_severe: last_moderate_severe],
                                             severe_ds[prev_severe: last_severe]])
            prev_none = last_none
            prev_mild = last_mild
            prev_moderate = last_moderate
            prev_moderate_severe = last_moderate_severe
            prev_severe = last_severe
        return eq_ds["d70"], eq_ds["d14"], eq_ds["d16"]
