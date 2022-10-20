import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.system('cls') if os.name == 'nt' else os.system('clear')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

x = [1,2,3,4,5,6,7,8]
print([0:10])

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
