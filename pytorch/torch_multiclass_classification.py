import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from helper_functions import plot_decision_boundary
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import os

os.system('clear')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device = }')

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 101

X_blob, y_blob = make_blobs(
    n_samples=1000,
    centers=NUM_CLASSES,
    cluster_std=1.5,
    random_state=RANDOM_SEED
)

m = 1000
r = 32*np.pi
p = 2*np.pi/NUM_CLASSES

X_blob = np.zeros((m*NUM_CLASSES,2))
y_blob = np.zeros(m*NUM_CLASSES)

for c in range(NUM_CLASSES):
    for i in range(100):
        x = (i/20)*np.cos(r/m*i + 0.1*(-1)**i + c*p)
        y = (i/20)*np.sin(r/m*i + c*p)
        
        X_blob[i + c*m] = x, y
        y_blob[i + c*m] = c 


# to(device) + train/test split
data = {
    s: l.to(device) for s, l in zip(
        ['X train', 'X test', 'y train', 'y test'],
        train_test_split(
            torch.from_numpy(X_blob).type(torch.float),
            torch.from_numpy(y_blob).type(torch.float),
            test_size=0.2,
            random_state=RANDOM_SEED)
    )
}


data['y train'] = data['y train'].type(torch.long)
data['y test'] = data['y test'].type(torch.long)

print(data['y train'])


plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()


class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=16):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model_4 = BlobModel(
    input_features=2,
    output_features=NUM_CLASSES,
    hidden_units=32
).to(device)

'''
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(data['X test'])
print(y_logits[:10])

print(data['y test'])'''


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

loss_fn = nn.CrossEntropyLoss()
optmizer = torch.optim.SGD(params=model_4.parameters(), lr=0.05)

epochs = 20000
for epoch in range(epochs):
    model_4.train()

    y_logits = model_4(data['X train'])
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, data['y train'])
    acc = accuracy_fn(y_true=data['y train'], y_pred=y_pred)

    optmizer.zero_grad()
    loss.backward()
    optmizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(data['X test'])
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, data['y test'])
        test_acc = accuracy_fn(y_true=data['y test'], y_pred=test_preds)

    if not epoch % 100:
        print(f'{epoch = }, {loss = :.2f}, {acc = :.2f}, {test_loss = :.2f}, {test_acc = :.2f}%')

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title('train', figure=fig)
        plot_decision_boundary(fig, model_4, data['X train'], data['y train'])
        plt.subplot(1, 2, 2)
        plt.title('Test', figure=fig)
        plot_decision_boundary(fig, model_4, data['X test'], data['y test'])
        fig.canvas.draw()

        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 1000, 500)
        cv2.imshow("output", img)

        if epoch >= epochs - 5:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)