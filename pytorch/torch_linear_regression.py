import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import platform
import os


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(
            in_features=1,
            out_features=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


def plot_predictions(
    train_data,
    train_labels,
    test_data,
    test_labels,
    predictions=None
):

    plt.figure(figsize=(10, 7))
    plt.scatter(
        train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(
        test_data, test_labels, c='g', s=4, label='Testing data')

    if predictions is not None:
        plt.scatter(
            test_data, predictions, c='r', s=4, label='Predictions')

    plt.legend(prop={'size': 14})
    plt.show()


def train_model(model):
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    torch.manual_seed(42)

    epochs = 200
    for epoch in range(epochs):
        model.train()

        # 1. Forward pass
        y_pred = model(X_train)
        # 2. Calculate the loss
        loss = loss_fn(y_pred, y_train)
        # 3. Optmizer zero grad
        optimizer.zero_grad()
        # 4. Perform backpropagation
        loss.backward()
        # 5. Optimizer step
        optimizer.step()

        # Testing
        model_1.eval()

        with torch.inference_mode():
            test_pred = model_1(X_test)
            test_loss = loss_fn(test_pred, y_test)

        if not epoch % 10:
            print(f'Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}')


def test_model(model):
    model.eval()
    with torch.inference_mode():
        y_preds = model(X_test)

    # print(y_preds)
    plot_predictions(
        train_data=X_train.cpu(),
        train_labels=y_train.cpu(),
        test_data=X_test.cpu(),
        test_labels=y_test.cpu(),
        predictions=y_preds.cpu()
    )


if __name__ == '__main__':

    clean_terminal = 'clear' if platform.system().lower() == 'linux' else 'cls'
    os.system(clean_terminal)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    weight = 0.7
    bias = 0.3

    X = torch.arange(start=0, end=1, step=0.02).unsqueeze(dim=1).to(device)
    y = (weight * X + bias)

    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]

    torch.manual_seed(42)
    model_1 = LinearRegressionModel().to(device)

    train_model(model_1)
    test_model(model_1)

    # saving model
    MODEL_PATH = pathlib.Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_NAME = 'pytorch_workflow_model_1.pth'
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
    torch.save(obj=model_1.state_dict(), f=MODEL_SAVE_PATH)

    # loading model
    loaded_model_1 = LinearRegressionModel().to(device)
    loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

    test_model(loaded_model_1)
