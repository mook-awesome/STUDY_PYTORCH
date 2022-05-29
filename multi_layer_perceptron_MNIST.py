import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torch

mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
mnist.target = mnist.target.astype(np.int8)
x = mnist.data /255
y = mnist.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=0)

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

ds_train = TensorDataset(X_train, y_train)
ds_test = TensorDataset(X_test, y_test)

loader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=64, shuffle=False)

model = nn.Sequential()
model.add_module('fc1', nn.Linear(28*28*1, 100))
model.add_module('relu1', nn.ReLU())
model.add_module('fc2', nn.Linear(100, 100))
model.add_module('relu2', nn.ReLU())
model.add_module('fc3', nn.Linear(100, 10))
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    print("epoch{} : 완료\n".format(epoch))

def test():
    model.eval()
    correct = 0
    with torch.no_grad(): # 추론과정에서는 미분이 필요하지 않음.
        for data, targets in loader_test:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) # 확률이 가장 농픈 레이블 선택
            correct  += predicted.eq(targets.data.view_as(predicted)).sum() # 정답과 일치한 경우 정답 카운트 증가
    # 정확도 출력
    data_num = len(loader_test.dataset)
    print("\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n".format(correct, data_num, 100*correct/data_num))

for epoch in range(3):
    train(epoch)

test()

index = 2018
model.eval() # 신경망을 추론 모드로 변경
data = X_test[index]
output = model(data)
_, predicted = torch.max(output.data, 0)
print("예측 결과 : {}".format(predicted))
X_test_show = (X_test[index]).numpy()
plt.imshow(X_test_show.reshape(28, 28), cmap='gray')
print("이 이미지 데이터의 정답 레이블은 {:.0f}입니다.".format(y_test[index]))