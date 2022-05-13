from numpy import number
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
Ver = 1

if Ver == 1:
    w1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(1, requires_grad=True)
    w3 = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)



    def forward(x1, x2, x3):
        # Definition of hypothesis

        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 +b
        return hypothesis

    def loss(x1, x2, x3, y):
        # Calculate cost 
        y_pred = forward(x1, x2, x3)
        cost = torch.mean((y_pred - y)**2)
        return cost       




    # Training data
    x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
    x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
    x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
    y_train  = torch.FloatTensor([[152], [185], [180], [196], [142]])

    # Linear regression
    # lr = learning rate
    optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
    # The number of repetation
    number_of_epochs = 1999
    for epoch in range(number_of_epochs + 1):
        cost = loss(x1_train, x2_train, x3_train, y_train) 
        
        # Initialize difference value of optimizer
        #   - Because pytorch cumulate differential value, should be initialzied.
        optimizer.zero_grad()

        # Calculate gradient
        cost.backward()
        
        # Update values W and b
        optimizer.step()

        if epoch % 100 ==0:
            print('Epoch {:4d}/{} W1: {:.3f}, W2: {:.3f}, W3: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                epoch, number_of_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
            ))
            # print("Epoch "+str(epoch) + "w: " +str(W.item()))


elif Ver ==2:
    x_train  =  torch.FloatTensor([[73,  80,  75], 
                                [93,  88,  93], 
                                [89,  91,  80], 
                                [96,  98,  100],   
                                [73,  66,  70]])  
    print(x_train.shape)
    y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
    
    w = torch.zeros((3,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    
    optimizer = optim.SGD([w, b], lr = 3e-5)

    number_of_epochs = 1999
    for epoch in range(number_of_epochs + 1):
        hypothesis = x_train.matmul(w) + b
        cost = torch.mean((hypothesis - y_train)**2)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if epoch % 100 ==0:
            # print("Epoch {:4}/{} hypotehsis: {} cost: {:6f}".format(epoch, number_of_epochs, hypothesis.squeeze().detach(), cost.item()))
            print("Epoch {:4}/{} W: {} cost: {:6f}".format(epoch, number_of_epochs, w.squeeze().detach(), cost.item()))
