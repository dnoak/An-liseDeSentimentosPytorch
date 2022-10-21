from string import punctuation
import numpy as np
import torch
import pandas as pd
import timeit
import json
import re
import os

class Text2Tensor:
    def __init__(
        self,
        dict_path: str = 'data/ptbr_dict_most_common.json',
        dataset_path: str = 'data/imdb-reviews-pt-br.csv',
        N_INPUTS: int = 10,
        MAX_TEXT_SIZE: int = 256
        ):
        '''
        Carrega o dicionário das palavras mais comuns do 
        portguês brasileiro, e carrega os reviews e labels do
        imdb na quantidade `N_INPUTS`
        '''
        self.N_INPUTS = N_INPUTS
        self.MAX_TEXT_SIZE = MAX_TEXT_SIZE
        self.X = torch.zeros(
            (self.N_INPUTS, MAX_TEXT_SIZE), dtype=torch.long
        )
        self.y = torch.zeros(
            (self.N_INPUTS), dtype=torch.long
        )

        with open(dict_path, 'r') as f:
            self.ptbr_dict = json.load(f)

        with open(dataset_path, 'r', encoding='utf-8') as f:
            imdb_reviews_load = pd.read_csv(f)
            imdb_reviews_pos = imdb_reviews_load[:+N_INPUTS//2]
            imdb_reviews_neg = imdb_reviews_load[-N_INPUTS//2:]
            self.imdb_reviews = pd.concat([imdb_reviews_pos, imdb_reviews_neg])

    def _timer_decorator(func):
        '''Timer'''
        def timer(self):
            ti = timeit.default_timer()
            func(self)
            tf = timeit.default_timer() - ti
            str_print = f'| Fn: {func.__name__} - {tf:.4f}s |'
            print(' '+'_'*(len(str_print)-2))
            print(str_print)
            print(' '+'‾'*(len(str_print)-2))
        return timer
    
    def remove_symbols(self, txt):
        all_words = []
        for t in txt:
            all_words += re.sub(r'[^\w\s]', '', t).lower().split()
        return all_words

    def txt2tensor(self, X_str, y_str):
        '''
        blabla
        '''
        X_tensor = torch.zeros(self.MAX_TEXT_SIZE, dtype=torch.long)

        X_str = ' '.join(self.remove_symbols([X_str]))

        X_str_split = X_str.split()
        if len(X_str_split) > self.MAX_TEXT_SIZE:
            X_str_split = X_str_split[0:self.MAX_TEXT_SIZE]
        
        for pos, palavra in enumerate(X_str_split):
            try:
                X_tensor[pos] = self.ptbr_dict[palavra.lower()]
            except: pass

        if y_str == 'pos':
            return X_tensor, torch.tensor(1, dtype=torch.long)
        else:
            return X_tensor, torch.tensor(0, dtype=torch.long)

    @_timer_decorator
    def transform_all(self):
        random_pos = np.arange(self.N_INPUTS)
        np.random.shuffle(random_pos)
        for rpos, pos in zip(random_pos, range(self.N_INPUTS)):
            X_str = self.imdb_reviews.iloc[pos]['text_pt']
            y_str = self.imdb_reviews.iloc[pos]['sentiment']
            self.X[rpos], self.y[rpos] = self.txt2tensor(X_str, y_str)
        return self.X, self.y

    def tensor2txt(self, pos):
        ptbr_inv_dict = {v: k for k, v in self.ptbr_dict.items()}
        print('Original text:')
        print(self.imdb_reviews.iloc[pos]['text_pt'])
        print('-'*50)
        for i in self.X[pos]:
            try:
                print(ptbr_inv_dict[int(i)], end=' ')
            except:
                print('?', end=' ')
        print(f"({'-' if self.y[pos] else '+'})")

def main():
    os.system('cls') if os.name == 'nt' else os.system('clear')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    t2t = Text2Tensor()

    t2t.transform_all()
    t2t.tensor2txt(0)
    print(len(t2t.X), t2t.X.shape)
    print(t2t.y)

if __name__ == '__main__':
    main()