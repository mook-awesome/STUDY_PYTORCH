import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# for reproducibliity
random.seed(777)
torch.manual_seed(777)

# hyperparameters
training_epochs = 15
batch_size = 100

# MNINST dataset
mnist_train = dsets.MNIST(
    root='MNIST_data', 
    train=True,
    transform = transforms.ToTensor(),
    download=True
)
mnist_test = dsets.MNIST(
    root='MNIST_data', 
    train=False,
    transform = transforms.ToTensor(),
    download=True
)

# dataset loader
data_loader = DataLoader(
    dataset=mnist_train,
    batch_size=batch_size,
    shuffle=True,               # epoch마다 미니배치를 셔플함.
    drop_last=True              # 마지막 batch를 버림.
)

# MNIST data image of shape 28 * 28 = 784
linear = nn.Linear(784, 10, bias=True).to()

# Define optimizer
criterian = nn.CrossEntropyLoss().to() # Include softmax function
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:

        X = X.view(-1, 28 * 28).to()
        Y = Y.to()

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterian(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost/total_batch

    print('Epoch:', '%04d'%(epoch + 1), 'cost=','{:.9f}'.format(avg_cost))
print('Learning finished')
