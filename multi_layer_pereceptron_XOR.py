import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# for reproducibility
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

model = nn.Sequential(
    nn.Linear(2, 10, bias=True),    # 1 input layer, 10 hidden layer
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),   # 10 input layer, 10 hidden layer
    nn.Sigmoid(),
    nn.Linear(10, 10, bias=True),   # 10 input layer, 10 hidden layer
    nn.Sigmoid(),
    nn.Linear(10, 1, bias=True),    # 10 input layer, 1 output layer
    nn.Sigmoid(),
).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

cost_axis = []
epoch_axis = []
for epoch in range(10001):
    optimizer.zero_grad()
    hypothesis = model(x)

    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()
    cost_axis = np.append(cost_axis, cost.item())
    epoch_axis = np.append(epoch_axis, epoch)

    if epoch % 100 == 0:
        print(epoch, cost.item())

with torch.no_grad():
    hypothesis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print('모델의 출력값(hypothesis): \n', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(predicted): \n', predicted.detach().cpu().numpy())
    print('실제값(y): \n', y.detach().cpu().numpy())
    print('정확도(accuracy): \n', accuracy.item())

plt.figure('Cost')
plt.plot(epoch_axis, cost_axis)
plt.show()