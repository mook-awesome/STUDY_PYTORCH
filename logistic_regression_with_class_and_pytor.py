import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
# class of logistic regression model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))

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

model = BinaryClassifier()
optimizer = optim.SGD(model.parameters(), lr=1)
number_of_epochs = 1000
for epoch in range(number_of_epochs + 1):
    hypothesis = model(x_train)
    cost = F.binary_cross_entropy(hypothesis, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 10 == 0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item()/len(correct_prediction)
        print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(epoch, number_of_epochs, cost.item(), accuracy*100))

