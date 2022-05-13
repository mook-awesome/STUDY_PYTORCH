import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import variable


torch.manual_seed(1)            # 다시 실행해도 같은 결과를 나오게 함.
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
W = torch.zeros(1, requires_grad=True)  
b = torch.zeros(1, requires_grad=True)

def forward(x):
    # Definition of hypothesis

    hypothesis =  x * W + b
    return hypothesis

def loss(x, y):
    # Calculate cost 
    y_pred = forward(x)
    cost = torch.mean((y_pred - y)**2)
    return cost       

# Linear regression
# lr = learning rate
optimizer = optim.SGD([W, b], lr=0.01)
# The number of repetation
number_of_epochs = 1999
for epoch in range(number_of_epochs + 1):
    cost = loss(x_train, y_train) 
    
    # Initialize difference value of optimizer
    #   - Because pytorch cumulate differential value, should be initialzied.
    optimizer.zero_grad()

    # Calculate gradient
    cost.backward()
    
    # Update values W and b
    optimizer.step()

    if epoch % 100 ==0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, number_of_epochs, W.item(), b.item(), cost.item()
        ))
        # print("Epoch "+str(epoch) + "w: " +str(W.item()))

