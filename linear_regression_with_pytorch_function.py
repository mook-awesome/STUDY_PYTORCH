import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
# Arguiment is dimensions of input and output
model = nn.Linear(1, 1)
print(list(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

number_of_epochs = 1999
for epoch in range(number_of_epochs):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 ==0:
        print("Epoch {:4d}/{} Cost: {:6f}".format(epoch, number_of_epochs, cost.item()))

new_var = torch.FloatTensor([4.0])
pred_y = model(new_var)
print("훈련 후 입력이 4일 때 예측값 : ", pred_y)