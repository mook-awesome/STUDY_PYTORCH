import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
# Training data
x_data = [
    [1, 2], 
    [2, 3], 
    [3, 1], 
    [4, 3], 
    [5, 3], 
    [6, 2]
]
y_data = [
    [0],
    [0],
    [0],
    [1],
    [1],
    [1]
]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)
model = nn.Sequential(
    nn.Linear(2, 1),       # W and b
    nn.Sigmoid()           # Output is affected by sigmoid
)
print(model(x_train))
optimizer = optim.SGD(model.parameters(), lr=1)
number_of_epochs = 1000
for epoch in range(number_of_epochs + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)      # binary_cross_entropy는 sigmoid에 맞는 loss function --> 확률 이론 참고
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:6f} Accuracy {:2.2f}%'.format(epoch, number_of_epochs, cost.item(), accuracy*100))
print(model(x_train))
print('Trained model:', list(model.parameters()))