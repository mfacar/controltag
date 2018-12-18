import pandas as pd
import itertools
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from util.TextPreProcessor import TextPreProcessor


class Preprocessor:

    def __init__(self, path):
        self.tokenizer = Tokenizer()
        self.text_pre_processor = TextPreProcessor()
        self.path = path

    def prepare_dataset_to_model(self, all_participants: pd.DataFrame):
        all_participants_mix = all_participants.copy()
        windows_size = 10

        all_participants_mix['answer'] = all_participants_mix.apply(
            lambda row: self.text_pre_processor.text_to_wordlist(row.answer).split(), axis=1)
        print(all_participants_mix.head())

        words = set(itertools.chain(*[w for w in all_participants_mix['answer']]))
        vocab_size = len(words)
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.tokenizer.fit_on_texts(all_participants_mix['answer'])

        with open(self.path + 'tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        all_participants_mix['t_answer'] = self.tokenizer.texts_to_sequences(all_participants_mix['answer'])

        return self.__generate_model_dataset(all_participants_mix, windows_size)

    @staticmethod
    def __generate_model_dataset(all_participants_mix: pd.DataFrame, windows_size):
        cont = 0
        phrases_lp = pd.DataFrame(columns=['personId', 'answer', 't_answer'])
        answers = all_participants_mix.groupby('personId').agg('sum', axis=1)

        for p in answers.iterrows():
            words = p[1]["answer"]
            size = len(words)
            word_tokens = p[1]["t_answer"]

            for i in range(size):
                if i + windows_size <= size:
                    sentence = words[i:min(i + windows_size, size)]
                    tokens = word_tokens[i:min(i + windows_size, size)]
                    phrases_lp.loc[cont] = [p[0], sentence, tokens]
                    cont = cont + 1

        phrases_lp["t_answer"] = pad_sequences(phrases_lp["t_answer"], value=0, padding="post",
                                               maxlen=windows_size).tolist()

        return phrases_lp
