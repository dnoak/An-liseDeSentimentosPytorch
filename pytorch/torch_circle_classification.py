import torch
from torch import nn
import requests
from helper_functions import plot_predictions, plot_decision_boundary
from pathlib import Path
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2

os.system('clear')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device = }')

plt.rcParams.update({'figure.max_open_warning': 1})

n_samples = 1000
X, y = make_circles(n_samples, noise=0.03, random_state=42)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

radius = 4*np.pi
phase = np.pi
samples = 100

m = 100
r = 4*np.pi
p = np.pi

X = np.zeros((200,2))
y = np.zeros(200)

for c in range(2):
    for i in range(100):
        x = (i/20)*np.cos(r/m*i + 0.1*(-1)**i + c*p)
        _y = (i/20)*np.sin(r/m*i + c*p)
        
        X[i + c*m] = x, _y
        y[i + c*m] = c 


circles = pd.DataFrame({
    'X1': X[:, 0],
    'X2': X[:, 1],
    'label': y
})

plt.scatter(
    x=X[:, 0],
    y=X[:, 1],
    c=y,
    cmap=plt.cm.RdYlBu
)
# plt.show()

data = {
    s: torch.tensor(l).type(torch.float).to(device) for s, l in zip(
        ['X train', 'X test', 'y train', 'y test'],
        train_test_split(X, y, test_size=0.2, random_state=42)
        )
    }


class CircleModelV2(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=16):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_3 = CircleModelV2(
    input_features=2,
    output_features=1,
    hidden_units=64
).to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc


epochs = 5000
loss_fn = nn.BCEWithLogitsLoss()
optmizer = torch.optim.SGD(model_3.parameters(), lr=0.1)

for epoch in range(epochs):
    model_3.train()
    y_logits = model_3(data['X train']).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, data['y train'])
    acc = accuracy_fn(y_true=data['y train'], y_pred=y_pred)
    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(data['X test']).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, data['y test'])
        test_acc = accuracy_fn(y_true=data['y test'], y_pred=test_pred)

    if not epoch % 100 == 0:
        print(f'{epoch = }, {loss = :.3f}, {acc = :.2f}%, {test_loss = :.2f}, {test_acc = :.2f}%')

    if not epoch % 10:
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('train', figure=fig)
        plot_decision_boundary(fig, model_3, data['X train'], data['y train'])
        plt.subplot(1, 2, 2)
        plt.title('Test', figure=fig)
        plot_decision_boundary(fig, model_3, data['X test'], data['y test'])
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 1000, 500)
        cv2.imshow("output", img)

        if epoch >= epochs - 10:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)








'''
for epoch in range(epochs):
    model_0.train()

    # 1. Forward pass
    y_logits = model_0(data['X train']).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    # 2. Loss
    loss = loss_fn(y_logits, data['y train'])
    acc = accuracy_fn(y_true=data['y train'], y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. gradient descent
    optimizer.step()

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(data['X test']).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, data['y test'])
        test_acc = accuracy_fn(y_true=data['y test'], y_pred=test_pred)

    if not epoch % 10:
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('train', figure=fig)
        plot_decision_boundary(fig, model_0, data['X train'], data['y train'])
        plt.subplot(1, 2, 2)
        plt.title('Test', figure=fig)
        plot_decision_boundary(fig, model_0, data['X test'], data['y test'])
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Create window with freedom of dimensions
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        # Resize window to specified dimensions
        cv2.resizeWindow("output", 1000, 500)
        cv2.imshow("output", img)
        if epoch >= epochs - 10:                  # Show image
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

    if not epoch % 10:
        print(
            f'{epoch = }, {loss = :.3f}, {acc = :.2f}%, {test_loss = :.2f}, {test_acc = :.2f}%')


if Path('helper_functions.py').is_file():
    print('already exists')
else:
    print('Downloading...')
    request = requests.get(
        'https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py')
    with open('helper_functions.py', 'wb') as f:
        f.write(request.content)
'''