import numpy as np
import torch
import pandas as pd
import timeit
import json
from functools import wraps
import re
import os

class Text2Tensor:
    def __init__(
        self,
        dict_path: str,
        dataset_path: str,
        X_col: str,
        y_col: str,
        n_text_inputs: int,
        max_text_size: int
        ):
        '''
        Carrega o dicionário das N palavras mais comuns do 
        do dataset e carrega as entradas e labels do dataset
        na quantidade `n_text_inputs`
        '''
        self.dict_path = dict_path
        self.dataset_path = dataset_path
        self.X_col = X_col
        self.y_col = y_col
        self.n_text_inputs = n_text_inputs
        self.max_text_size = max_text_size

        self.X = torch.zeros((self.n_text_inputs, self.max_text_size), dtype=torch.long)
        self.y = torch.zeros((self.n_text_inputs), dtype=torch.long)

        self.dataset = pd.read_csv(self.dataset_path).sample(frac=1) #randomize rows

        with open(self.dict_path, 'r') as f:
            self.dictionary = json.load(f)

    def remove_symbols(self, text):
        all_words = []
        for t in text:
            all_words += re.sub(r'[^\w\s]', '', t).lower().split()
        return all_words

    def text2tensor(self, X_text, y_text):
        '''
        blabla
        '''
        X_words = ' '.join(self.remove_symbols([X_text])).split()[0:self.max_text_size]

        # Sem descontinuação de palavras
        # X_ids = [self.dictionary.get(w, 0) for w, _ in zip(X_words, range(self.max_text_size))]
        # X_ids_no_zeros = [id for id in X_ids if id]
        # X_ids_no_zeros_max_size = X_ids_no_zeros + [0]*(self.max_text_size - len(X_ids_no_zeros))
        
        # Palavras contínuas
        X_ids = [self.dictionary.get(w, 0) for w, _ in zip(X_words, range(self.max_text_size))]
        X_ids_max_size = X_ids + [0]*(self.max_text_size - len(X_ids))

        X_tensor = torch.tensor(X_ids_max_size, dtype=torch.long)
        y_tensor = torch.tensor(1, dtype=torch.long) if y_text == 'pos' else torch.tensor(0, dtype=torch.long)

        return X_tensor, y_tensor

    def _timer_decorator(func):
        @wraps(func)
        def timer(*args, **kwargs):
            ti = timeit.default_timer()
            print()
            func(*args, **kwargs)
            tf = timeit.default_timer() - ti
            str_print = f'| Fn: {func.__name__} - {tf:.4f}s |'
            print(' '+'_'*(len(str_print)-2))
            print(str_print)
            print(' '+'‾'*(len(str_print)-2))
        return timer

    @_timer_decorator
    def transform_all(self):
        for pos in range(self.n_text_inputs):
            X_str = self.dataset.iloc[pos][self.X_col]
            y_str = self.dataset.iloc[pos][self.y_col]
            self.X[pos], self.y[pos] = self.text2tensor(X_str, y_str)
        return self.X, self.y

    def tensor2text(self, pos):
        inverse_dict = {v: k for k, v in self.dictionary.items()}
        print('Original text:')
        print(self.dataset.iloc[pos][[self.X_col, self.y_col]])
        print('-'*100)

        for word in self.X[pos]:
            try:
                print(inverse_dict[int(word)], end=' ')
            except:
                print('?', end=' ')
        print(f"({'-' if self.y[pos] == 0 else '+'})")

def main():
    os.system('cls') if os.name == 'nt' else os.system('clear')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    t2t = Text2Tensor(
        dataset_path='data/imdb/imdb-reviews-pt-br.csv',
        dict_path='data/ptbr/ptbr_imdb_5000.json',
        X_col='text_pt',
        y_col='sentiment',
        n_text_inputs=100,
        max_text_size=256
    )

    t2t.transform_all()
    t2t.tensor2text(pos=3)

    print(len(t2t.X), t2t.X.shape)
    print(t2t.y)

if __name__ == '__main__':
    main()