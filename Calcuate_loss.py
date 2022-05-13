import numpy as np
import matplotlib.pyplot as plt

w = 1.0             # a random guess : random value
x_data = [1,2,3]
y_data = [2,4,6]

# output omdel for the forward pass
def forward(x):
    return x * w

# loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_axis = np.arange(0.0, 4.1, 0.1)
MSE = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=",w)
    l_sum=0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        l = loss(x_val, y_val)
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    MSE = np.append(MSE, l_sum/3)
    print("MSE=", l_sum/3)

plt.figure('Loss')
plt.xlabel('w value')
plt.ylabel('Loss')
plt.plot(w_axis, MSE)
plt.show()