import keras
import pandas as pd


class DataReader:
    BINS = [-1, 0, 5, 10, 15, 25]
    LABELS = [0, 1, 2, 3, 4]
    NUM_CLASSES = 5

    all_participants: pd.DataFrame
    ds_total: pd.DataFrame
    ds_balanced: pd.DataFrame
    num_classes: int

    def __init__(self, path='/Users/mercyfalconi/PycharmProjects/ControlTAC/data/'):
        self.read_transcript_dataset(path)
        self.load_avec_datasets(path)

    def read_transcript_dataset(self, path):
        self.all_participants = pd.read_csv(path + 'all.csv', sep=',')
        self.all_participants.columns = ['index', 'personId', 'question', 'answer']
        self.all_participants = self.all_participants.astype({"index": float, "personId": float, 'question': str,
                                                              'answer': str})

    def load_avec_dataset_file(self, path, score_column):
        ds = pd.read_csv(path, sep=',')
        ds['level'] = pd.cut(ds[score_column], bins=self.BINS, labels=self.LABELS)
        ds['PHQ8_Score'] = ds[score_column]
        ds['cat_level'] = keras.utils.to_categorical(ds['level'], self.NUM_CLASSES).tolist()
        ds = ds[['Participant_ID', 'level', 'cat_level', 'PHQ8_Score']]
        ds = ds.astype({"Participant_ID": float, "level": int, 'PHQ8_Score': int})
        return ds

    def load_avec_datasets(self, path):
        train = self.load_avec_dataset_file(path + 'train_split_Depression_AVEC2017.csv', 'PHQ8_Score')
        dev = self.load_avec_dataset_file(path + 'dev_split_Depression_AVEC2017.csv', 'PHQ8_Score')
        test = self.load_avec_dataset_file(path + 'full_test_split.csv', 'PHQ_Score')
        print("Longitud: entrenamiento= {}, validacion= {}, test= {}".format(len(train), len(dev),
                                                                             len(test)))
        self.ds_total = pd.concat([train, dev, test])
        total_phq8 = len(self.ds_total)
        print("Longitud total = {}".format(total_phq8))
        self.balanced_ds()

    def check_phq_target_distribution(self):
        ds = self.ds_total
        none_ds = ds[ds['level'] == 0]
        mild_ds = ds[ds['level'] == 1]
        moderate_ds = ds[ds['level'] == 2]
        moderate_severe_ds = ds[ds['level'] == 3]
        severe_ds = ds[ds['level'] == 4]

        print("Cantidad por none_ds: {}, mild_ds: {}, moderate_ds {}, moderate_severe_ds: {}, severe_ds {}".format(
            len(none_ds), len(mild_ds), len(moderate_ds), len(moderate_severe_ds), len(severe_ds)))

    @staticmethod
    def split_by_phq_level(ds):
        none_ds = ds[ds['level'] == 0]
        mild_ds = ds[ds['level'] == 1]
        moderate_ds = ds[ds['level'] == 2]
        moderate_severe_ds = ds[ds['level'] == 3]
        severe_ds = ds[ds['level'] == 4]
        return none_ds, mild_ds, moderate_ds, moderate_severe_ds, severe_ds

    def balanced_ds(self):
        ds_total = self.ds_total
        b_none_ds = ds_total[ds_total['level'] == 0]
        b_mild_ds = ds_total[ds_total['level'] == 1].sample(26)
        b_moderate_ds = ds_total[ds_total['level'] == 2].sample(26)
        b_moderate_severe_ds = ds_total[ds_total['level'] == 3]
        b_severe_ds = ds_total[ds_total['level'] == 4]

        self.ds_balanced = pd.concat([b_none_ds, b_mild_ds, b_moderate_ds, b_moderate_severe_ds, b_severe_ds])
