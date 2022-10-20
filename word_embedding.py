from most_common_words_dict import MostCommonWordsDict
from txt2tensor import Text2Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

os.system('cls') if os.name == 'nt' else os.system('clear')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f' -- Using {device} --')

MCWD = MostCommonWordsDict(VOCAB_SIZE=500)
ptbr_dict = MCWD.generate_dict()

T2T = Text2Tensor(N_INPUTS=500)
T2T.transform_all()

X_text, y_text = T2T.X, T2T.y

data = {
    s: l for s, l in zip(
        ['X train', 'X test', 'y train', 'y test'],
        train_test_split(
            X_text.to(device),y_text.to(device),
            test_size=0.2,
            random_state=1
        )
    )
}

[print(d,'-', len(data[d])) for d in data]

CONTEXT_SIZE = 2
EMBEDDING_DIM = 20
VOCAB_SIZE = len(ptbr_dict)+1

train_data = TensorDataset(data['X train'], data['y train'])
test_data = TensorDataset(data['X test'], data['y test'])

data_loader = {
    'train': DataLoader(train_data, batch_size=50, shuffle=True),
    'test': DataLoader(train_data, batch_size=50, shuffle=True)
}

class SentimentalAnalysisNet(nn.Module):
    '''
    dadsdasda
    '''
    def __init__(
        self,
        input_features: int, 
        output_features: int, 
        embeding_dim: int,
        hidden_units: int = 16
        ):
        '''
        dasdasdasd
        '''
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_units = hidden_units
        
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, hidden_units, )