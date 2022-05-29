import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import torch
import torch.nn as nn
from torch import optim

digits = load_digits() 

print('image[0]: \n{}'.format(digits.images[0]))
print('target[0]: \n{}'.format(digits.target[0]))
print('전체 샘플의 수: {}'.format(len(digits.images)))

images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:5]):  # 5개의 샘플만 출력
#     plt.subplot(2, 5, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('sample: %i' %label)
# for i in range(5):
#     print(i, "번 인텍스 샘플의 레이블 : ", digits.target[i])
# plt.show()

X = digits.data
Y = digits.target

model = nn.Sequential(
    nn.Linear(64, 32),  # Input layer = 64, Hidden layer 1 = 32
    nn.ReLU(),          
    nn.Linear(32,16),   # Hidden layer 2 = 32, Hidden layer 3 = 16
    nn.ReLU(),  
    nn.Linear(16, 10)   # Hidden layer 3 = 16, Ouput layer = 10
)

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters()) # ReLU에 적함한 optimizer

losses = []
for epoch in range(100):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    loss.backward()
    optimizer.step()

    if epoch % 10 ==0:
        print('Epoch {:4d}/{} Cost : {:.6f}'.format(epoch, 100, loss.item()))
    losses.append(loss.item())
plt.plot(losses)
plt.show()