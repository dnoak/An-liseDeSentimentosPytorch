from concurrent.futures.process import _process_chunk
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
        vocab_size: int = 10,
        dataset_path: str = 'data/imdb-reviews-pt-br.csv'
        ):
        '''
        blabla
        '''
        self.vocab_size = vocab_size
        self.ptbr_dict = {}
        self.txt_inputs = pd.read_csv(dataset_path)[0:self.vocab_size]

    def remove_symbols(self, texts):
        all_words = []
        for t in texts:
            all_words += re.sub(r'[^\w\s]', '', t).lower().split()
        return all_words

    def generate_dict(self):
        
        all_words = self.remove_symbols(self.txt_inputs['text_pt'])

        count_words = Counter(all_words)
        sorted_words = count_words.most_common(self.vocab_size)
        self.ptbr_dict = { word[0]: i+1 for i, word in enumerate(sorted_words) }

        return self.ptbr_dict

    def save_dict(self):
        with open('data/ptbr_dict_most_common.json', 'w') as f:
            json.dump(self.ptbr_dict, f)

def main():
    os.system('cls') if os.name == 'nt' else os.system('clear')

    A = MostCommonWordsDict(vocab_size=500)
    words_dict = A.generate_dict()
    A.save_dict()
    [print(f'{i}: {words_dict[i]}') for i, _ in zip(words_dict, range(30))]

    print(f'{len(words_dict) = }')

if __name__ == '__main__':
    main()