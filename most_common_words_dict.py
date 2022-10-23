import json
from string import punctuation
from collections import Counter
from timeit import default_timer as timer
import pandas as pd
import re
import os

class MostCommonWordsDict():
    def __init__(
        self,
        dataset_path: str,
        vocab_size: int,
        column: str = None,
        ):
        '''
        blabla
        '''
        self.column = column
        self.vocab_size = vocab_size
        self.dataset_path = dataset_path
        self.dict = {}
        self.dict_path = f'data/ptbr/ptbr_{self.dataset_path.split("/")[-2]}_{self.vocab_size}.json'

    def load_dict(self):
        try:
            with open(self.dict_path, 'r') as f:
                return json.load(f)
        except ValueError as e :
            print(e)

    def remove_symbols(self, texts):
        all_words = []
        for t in texts:
            all_words += re.sub(r'[^\w\s]', '', t).lower().split()
        return all_words

    def generate_dict(self, save_dict = False):
        data = pd.read_csv(self.dataset_path)[0:self.vocab_size]
        all_words = self.remove_symbols(data[self.column])

        count_words = Counter(all_words)
        sorted_words = count_words.most_common(self.vocab_size)
        self.dict = { word[0]: i+1 for i, word in enumerate(sorted_words) }

        if save_dict:
            with open(self.dict_path, 'w') as f:
                json.dump(self.dict, f)

        return self.dict

def main():
    os.system('cls') if os.name == 'nt' else os.system('clear')

    A = MostCommonWordsDict(
        column='text_pt',
        dataset_path='data/imdb/imdb-reviews-pt-br.csv', #'data\wikipedia\wiki_imdb_summaries.csv',
        vocab_size=50,
    )
    words_dict = A.generate_dict(save_dict=True)
    print(len(A.load_dict()))

    #[print(f'{i}: {words_dict[i]}') for i, _ in zip(words_dict, range(30))]

    #print(f'{len(words_dict) = }')

if __name__ == '__main__':
    main()