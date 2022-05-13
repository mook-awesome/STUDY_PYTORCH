import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

x_train = torch.FloatTensor([   [73, 80, 75], 
                                [93, 88, 93], 
                                [89, 91, 90],
                                [96, 98, 100],
                                [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
model = nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
number_of_epochs = 1999
for epoch in range(number_of_epochs + 1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:6f}'.format(epoch, number_of_epochs, cost.item()))

new_var = torch.FloatTensor([73, 80, 75])
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일 때의 예측값 : ', pred_y)