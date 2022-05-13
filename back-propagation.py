import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import variable

# torch.cuda.get_device_name(0)
# print(torch.cuda.is_available())

# output omdel for the forward pass
def forward(x):
    return x * w

# loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# gradient function
def gradient(x, y):
    return 2 * x * (x * w -y)

# Main function --------------------------------------
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
# w = 1.0

w = variable(torch.Tensor([1.0]), requires_grad=True) # Any random value
# requires_grad : save slope value (differenciate)

# x = variable(torch.randn(1, 10))
# prev_h = variable(torch.randn(1, 20))
# w_h = variable(torch.randn(20,20))
# w_x = variable(torch.randn(20, 10))
# i2h = torch.mm(w_x, x.t())
# h2h = torch.mm(w_h, prev_h.t())
# next_h = i2h + h2h
# next_h = next_h.tanh()
# next_h.backward(torch.ones(1, 20))

for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()     # calculate slope   
        print('\tgrad: ', x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data
        # manualyy zero the gradients after updating weights
        w.grad.data.zero_()
                
    print("progress: ", epoch, l.data[0])
# after training
print("predict (after training)", "4 hours", forward(4).data[0])