import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GRAPH = False

# Sigmoid function ------------------------------------------
def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
w = [0.5, 1.0, 1.5]
b = [0.5, 1.0, 1.5]
y1 = sigmoid(w[0] * x)
y2 = sigmoid(w[1] * x)
y3 = sigmoid(w[2] * x)

y_with_b1 = sigmoid(b[0] + x)
y_with_b2 = sigmoid(b[1] + x)
y_with_b3 = sigmoid(b[2] + x)

if GRAPH == True:
    plt.figure('For w variation')
    plt.plot(x, y1, 'r', linestyle='--', label='w='+str(w[0]))
    plt.plot(x, y2, 'g',                 label='w='+str(w[1]))
    plt.plot(x, y3, 'b', linestyle='--', label='w='+str(w[2]))
    plt.plot([0, 0], [1.0, 0.0], ':')
    plt.title('Sigmoid Function with w variation')
    plt.legend()

    plt.figure('For b variation')
    plt.plot(x, y_with_b1, 'r', linestyle='--', label='b='+str(b[0]))
    plt.plot(x, y_with_b2, 'g',                 label='b='+str(b[1]))
    plt.plot(x, y_with_b3, 'b', linestyle='--', label='b='+str(b[2]))
    plt.plot([0, 0], [1.0, 0.0], ':')
    plt.title('Sigmoid Function with b variation')
    plt.legend()
    plt.show()
# /Sigmoid function -----------------------------------------

# Logistic regression ---------------------------------------
torch.manual_seed(1)        # To generate same result 
x_data = [  [1, 2], 
            [2, 3], 
            [3, 1], 
            [4, 3], 
            [5, 3], 
            [6, 2]]
y_data = [  [0], 
            [0],
            [0],
            [1],
            [1],
            [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
print(x_train.shape)
print(y_train.shape)

w = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1     , requires_grad=True)
print(w.detach().numpy()[0])
optimizer = optim.SGD([w, b], lr=1)         # lr = learning rate
number_of_epochs = 1000

for epoch in range(number_of_epochs + 1):

    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(w)+b)))
    losses = -(y_train*torch.log(hypothesis) + (1-y_train)*torch.log(1-hypothesis))
    cost = losses.mean()
    optimizer.zero_grad()                   # Set initial gradient value to zero
    cost.backward()                         # Calculate gradient
    optimizer.step()                        # update w, b  
    if epoch % 100 == 0:
        print('Epoch: {:4d}/{} Cost: {:.6f} w: {}, {} b: {:.4f}'.format(epoch, number_of_epochs, cost.item(), w.detach().numpy()[0], w.detach().numpy()[1], b.item()))
hypothesis = torch.sigmoid(x_train.matmul(w) + b)
print('hypothesis: ', hypothesis)
prediction = hypothesis >= torch.FloatTensor([0.5])
print('prediction: ', prediction)