from most_common_words_dict import MostCommonWordsDict
from text2tensor import Text2Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import copy
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f' ____Using {device}____')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

class SplitDataset():
    def __init__(
        self, dataset_path, X_col, y_col, batch_size, 
        dict_path, n_text_inputs, max_text_size,
        split_proportion = [[0, 0.6], [0.6, 0.8], [0.8, 1]],
        ):
        '''
        blablalbal
        '''
        self.dataset_path = dataset_path
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.dict_path = dict_path
        self.n_text_inputs = n_text_inputs
        self.max_text_size = max_text_size
        self.split_proportion = split_proportion

        self.dataset = {}
        self.dataloader = {}

    def load_dataset(self):
        ...

    def create_dataset(self):
        t2t = Text2Tensor(
            dict_path=self.dict_path,
            dataset_path=self.dataset_path,
            X_col=self.X_col,
            y_col=self.y_col,
            n_text_inputs=self.n_text_inputs,
            max_text_size=self.max_text_size
        )

        t2t.transform_all()
        X_text, y_text = t2t.X, t2t.y

        self.dataset = {
            **{
                X_ttv : X_text[int(p[0]*self.n_text_inputs):int(p[1]*self.n_text_inputs)] 
                for X_ttv, p in zip(
                    ['X train', 'X test', 'X validation'],
                    self.split_proportion)
            },

            **{
                y_ttv : y_text[int(p[0]*self.n_text_inputs):int(p[1]*self.n_text_inputs)]
                for y_ttv, p in zip(
                    ['y train', 'y test', 'y validation'],
                    self.split_proportion)
            },
        }
        return self.dataset
    
    def create_dataloader(self):
        self.dataloader = {
            'train': DataLoader(
                Dataset(self.dataset['X train'], self.dataset['y train']), 
                batch_size=self.batch_size,
                shuffle=True),

            'test': DataLoader(
                Dataset(self.dataset['X test'], self.dataset['y test']),
                batch_size=self.batch_size,
                shuffle=True),

            'validation': DataLoader(
                Dataset(self.dataset['X validation'], self.dataset['y validation']),
                batch_size=self.batch_size,
                shuffle=True),
        }
        return self.dataloader


class SentimentAnalysisNN(nn.Module):
    '''
    dadsdasda
    '''
    def __init__(
        self,
        device: torch.device,
        input_shape: int, 
        output_shape: int, 
        embedding_dim: int,
        hidden_size_lstm: int,
        num_layers_lstm: int = 2,
        dropout_lstm: float = 0.5,
        ):
        '''
        dasdasdasd
        '''
        super().__init__()
        self.device = device

        self.num_layers_lstm = num_layers_lstm
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size_lstm = hidden_size_lstm
        
        self.embedding = nn.Embedding(input_shape, embedding_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size_lstm, 
            num_layers = num_layers_lstm,
            batch_first = True,
            dropout = dropout_lstm
        )
        self.batch_normalization = nn.BatchNorm1d(hidden_size_lstm)

        self.dropout = nn.Dropout(0.3)

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(hidden_size_lstm, output_shape),
            nn.Sigmoid(),
            #nn.Linear(64, 16),
            #nn.Sigmoid(),
            #nn.Linear(16, output_shape),
            #nn.Sigmoid(),
        )

    def forward(self, x, hidden):
        #batch_size = batch_size #x.size()
        #print(1, batch_size)
        embedd = self.embedding(x)

        lstm_out, hidden = self.lstm(embedd, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size_lstm)
        x = self.batch_normalization(lstm_out)
        x = self.dropout(x)
        x = self.linear_layer_stack(x)
        x = x.view(batch_size, -1)
        x = x[:, -1]

        return x, hidden
    
    def init_hidden(self, batch_size):
        #print(2, batch_size)
        weight = next(self.parameters()).data

        hidden = (
            weight.new(
                self.num_layers_lstm, 
                batch_size, 
                self.hidden_size_lstm
                ).zero_().to(self.device),
            
            weight.new(
                self.num_layers_lstm, 
                batch_size, 
                self.hidden_size_lstm
                ).zero_().to(self.device)
            )
        return hidden

def save_model(epoch, loss, model_state_dict, optimizer_state_dict):
    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, global_model_path)

def accuracy_fn(y_true, y_pred):
    y_pred = torch.round(y_pred).long()
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(
    model: torch.nn.Module,
    epoch: int,
    batch_size: int,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_value: int,
    accuracy_fn,
    device: torch.device = device,
    ):
    
    train_loss, train_acc = 0, 0
    hidden = model.init_hidden(batch_size)

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        hidden = tuple([each.data for each in hidden])

        train_pred, hidden = model(X, hidden)
        
        loss = loss_fn(train_pred, y.float())
        train_loss += loss
        train_acc += accuracy_fn(y, train_pred)

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    print(f'\n{" "*15}epoch: {epoch}')
    print(f'train      | loss: {train_loss:.2f} | acc: {train_acc:.2f}%')

@torch.inference_mode()
def test_step(
    model: torch.nn.Module,
    tolerance: float,
    epoch: int,
    batch_size: int,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
    ):

    test_loss, test_acc = 0, 0
    hidden = model.init_hidden(batch_size)

    model.eval()
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)

        hidden = tuple([each.data for each in hidden])

        test_pred, hidden = model(X, hidden)

        test_loss += loss_fn(test_pred, y.float())
        test_acc += accuracy_fn(y, test_pred)
    
    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f'test       | loss: {test_loss:.2f} | acc: {test_acc:.2f}%')

    best_loss = torch.load(global_model_path)['loss']

    '''if test_loss < best_loss + tolerance: #- tolerance:
        save_model(
            epoch=epoch, 
            loss=test_loss,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            )'''
    '''save_model(
        epoch=epoch, 
        loss=test_loss,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        )'''


@torch.inference_mode()
def validation_step(
    model: torch.nn.Module,
    batch_size: int,
    validation_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
    ):

    #model = copy.deepcopy(model)
    validation_loss, validation_acc = 0, 0
    hidden = model.init_hidden(batch_size)
    model.eval()
    #with torch.inference_mode():
    for X, y in validation_dataloader:
        X, y = X.to(device), y.to(device)
        print(X.shape)

        hidden = tuple([each.data for each in hidden])

        validation_pred, hidden = model(X, hidden)

        validation_loss += loss_fn(validation_pred, y.float())
        validation_acc += accuracy_fn(y, validation_pred)

    validation_loss /= len(validation_dataloader)
    validation_acc /= len(validation_dataloader)

    print(f'validation | loss: {validation_loss:.2f} | acc: {validation_acc:.2f}%')
    print(f'{" "*15}sample:')
    print(f'{y.cpu().numpy()[0:20]}')
    print(f'{torch.round(validation_pred).long().cpu().numpy()[0:20]}\n')

def train_neural_network():
    for epoch in range(epochs):
        '''checkpoint = torch.load(global_model_path)
        best_loss = checkpoint['loss']

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])'''
        
        train_step(
            model=model,
            epoch=epoch,
            batch_size=batch_size,
            train_dataloader=dataloader['train'],
            loss_fn=loss_fn,
            optimizer=optimizer,
            clip_value=clip_value,
            accuracy_fn=accuracy_fn,
            device=device
        )

        test_step(
            model=model,
            tolerance=tolerance,
            epoch=epoch,
            batch_size=batch_size,
            test_dataloader=dataloader['test'],
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )

        validation_step(
            model=model,
            batch_size=batch_size,
            validation_dataloader=dataloader['validation'],
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

@torch.inference_mode()
def prediction_loop():
    t2t = Text2Tensor(
        dict_path=f'data\ptbr\ptbr_imdb_{vocab_size-1}.json',
        dataset_path='data\imdb\imdb-reviews-pt-br.csv',
        X_col='text_pt',
        y_col='sentiment',
        n_text_inputs=1,
        max_text_size=256
    )

    checkpoint = torch.load(global_model_path)
    best_loss = checkpoint['loss']

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    while True:

        text_tensor, _ = t2t.text2tensor(input('Text: '), 'pos')
        text_tensor = text_tensor.to(device).unsqueeze(dim=0)

        hidden = model.init_hidden(batch_size)
        hidden = tuple([each.data for each in hidden])

        validation_pred, hidden = model(text_tensor, hidden)
        if torch.round(validation_pred).int():
            print('Positivo')
        else:
            print('Negativo')


if __name__ == '__main__':
    os.system('cls') if os.name == 'nt' else os.system('clear')
    
    vocab_size = 500
    vocab_size += 1 # novo id (0) para palavras não existentes no vocabulário
    max_text_size = 256
    n_text_inputs = 400

    batch_size = 1

    embedding_dim = 100
    hidden_size_lstm = 64 # 64
    num_layers_lstm = 2 # 2
    output_shape = 1

    lr = 0.001
    clip_value = 5
    epochs = 1000
    tolerance = 0.001

    data = SplitDataset(
        dataset_path='data\imdb\imdb-reviews-pt-br.csv',
        X_col='text_pt',
        y_col='sentiment',
        batch_size=batch_size,
        dict_path=f'data\ptbr\ptbr_imdb_{vocab_size-1}.json',
        n_text_inputs=n_text_inputs,
        max_text_size=max_text_size,
    )

    dataset = data.create_dataset()
    dataloader = data.create_dataloader()

    model = SentimentAnalysisNN(
        device=device,
        input_shape=vocab_size,
        output_shape=output_shape,
        embedding_dim=embedding_dim,
        hidden_size_lstm=hidden_size_lstm,
    ).to(device)
    
    global_model_path = f'data/models/{type(model).__name__}.pt'

    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not os.path.exists(global_model_path):
        save_model(
            epoch=0,
            loss=100,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict()
            )
    
    #train_neural_network()
    prediction_loop()

    