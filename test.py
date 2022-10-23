import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import os

os.system('cls') if os.name == 'nt' else os.system('clear')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


x1  = torch.tensor([[1,2,3,4]])
x2 = torch.tensor([[1,2,3,4],[5,6,7,8]])

print(x1.shape, x2.shape)
print(x1.squeeze().shape, x2.squeeze().shape)