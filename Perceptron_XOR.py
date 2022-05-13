import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

x = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1, bias=True)
sigmoid = nn.Sigmoid()
# nn.Sequential은 순차적으로 실행하도록 담는 container를 의미
model = nn.Sequential(linear, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(x)

    cost = criterion(hypothesis, y)
    cost.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, cost.item())

with torch.no_grad():
    hypotheis = model(x)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == y).float().mean()
    print('모델의 출력값(hypothesis): \n', hypothesis.detach().cpu().numpy())
    print('모델의 예측값(predicted): \n', predicted.detach().cpu().numpy())
    print('실제값(y): \n', y.detach().cpu().numpy())
    print('정확도(accuracy): \n', accuracy.item())