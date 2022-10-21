from random import seed
from most_common_words_dict import MostCommonWordsDict
from txt2tensor import Text2Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import copy
import os

os.system('cls') if os.name == 'nt' else os.system('clear')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f' -- Using {device} --')

vocab_size = 10000
vocab_size += 1
n_inputs = 5000

batch_size = 20

embedding_dim = 200
hidden_size_lstm = 256 # 64
num_layers_lstm = 8 # 2
output_shape = 1

lr = 0.001
clip = 5
epochs = 1000

MCWD = MostCommonWordsDict(vocab_size)
ptbr_dict = MCWD.generate_dict()
T2T = Text2Tensor(N_INPUTS=int(n_inputs * 1.1))
T2T.transform_all()
X_text, y_text = T2T.X, T2T.y

data = {
    **{
        s: l for s, l in zip(
            ['X train', 'X test', 'y train', 'y test'],
            train_test_split(
                X_text[:int(n_inputs*0.9)].to(device),
                y_text[:int(n_inputs*0.9)].to(device),
                test_size=0.2,
                random_state=1
            )
        )
    },
    'X eval': X_text[int(-n_inputs*0.1):].to(device),  
    'y eval': y_text[int(-n_inputs*0.1):].to(device),
}

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

dataloader = {
    'train': DataLoader(
        Dataset(data['X train'], data['y train']), 
        batch_size=batch_size,
        shuffle=True
    ),
    'test': DataLoader(
        Dataset(data['X test'], data['y test']),
        batch_size=batch_size,
        shuffle=True
    ),
    'eval': DataLoader(
        Dataset(data['X eval'], data['y eval']),
        batch_size=batch_size,
        shuffle=True
    )
}

#[print(d,'-', len(data[d])) for d in data]


#train_data = TensorDataset(data['X train'], data['y train'])
#test_data = TensorDataset(data['X test'], data['y test'])


class SentimentalAnalysisNet(nn.Module):
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
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
        batch_size = x.size()

        embedd = self.embedding(x)
        #print(embedd.shape, hidden[0].shape)

        lstm_out, hidden = self.lstm(embedd, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_size_lstm)
        x = self.batch_normalization(lstm_out)
        #x = self.dropout(lstm_out)
        x = self.dropout(x)
        x = self.linear_layer_stack(x)
        x = x.view(batch_size, -1)
        x = x[:, -1]

        return x, hidden
    
    def init_hidden(self, batch_size):
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

def accuracy_fn(y_true, y_pred):
    y_pred = torch.round(y_pred).long()#torch.as_tensor((y_pred-0.5)>0, dtype=torch.long)
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def train_step(
    model: torch.nn.Module,
    epoch: int,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
    ):
    # global best_model
    
    # model = best_model['model']
    train_loss, train_acc = 0, 0
    hidden = model.init_hidden(batch_size)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        hidden = tuple([each.data for each in hidden])

        y_pred, hidden = model(X, hidden)
        
        loss = loss_fn(y_pred.squeeze(), y.float())
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred)

        optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(model_0.parameters(), clip)

        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    print(f'{epoch = }, {train_loss = :.2f}, {train_acc = :.2f}%,', end=' ')
    
    # if train_acc >= best_model['acc']:
    #     best_model = {'acc': train_acc, 'model': model}
    
def test_step(
    model: torch.nn.Module,
    epoch: int,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
    ):
    global best_model
    
    #model = best_model['model']

    test_loss, test_acc = 0, 0
    hidden = model.init_hidden(batch_size)
    model.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)

            hidden = tuple([each.data for each in hidden])

            test_pred, hidden = model(X, hidden)

            test_loss += loss_fn(test_pred.squeeze(), y.float())
            test_acc += accuracy_fn(y, test_pred)
        
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        
        print(f'{test_loss = :.2f}, {test_acc = :.2f}%')
        print(f'{y.cpu().numpy()[0:20]}\n{torch.round(test_pred).long().cpu().numpy()[0:20]}')
    
    # print(test_acc, best_model['acc'])
    # if test_acc >= best_model['acc'] - 0.2:
    #     model = copy.deepcopy(best_model['model'])
    #     best_model = {'acc': test_acc, 'model': copy.deepcopy(model)}

model_0 = SentimentalAnalysisNet(
    device=device,
    input_shape=vocab_size,
    output_shape=output_shape,
    embedding_dim=embedding_dim,
    hidden_size_lstm=hidden_size_lstm,
).to(device)

print(model_0)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model_0.parameters(), lr=lr)

best_model = {'acc': 0, 'model': model_0}
for epoch in range(epochs):
    train_step(
        model=model_0,
        epoch=epoch,
        train_dataloader=dataloader['train'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )

    test_step(
        model=model_0,
        epoch=epoch,
        test_dataloader=dataloader['test'],
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )