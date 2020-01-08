import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import load_fashion_mnist
from model.torch_model import torch_MLP


np.random.seed(19941017)

x_train, y_train, x_val, y_val = load_fashion_mnist('./data')


num_feature = x_train.shape[1]
total_num = len(x_train)
output_size = y_train.shape[1]
print(f"feature number : {num_feature} | data_number : {total_num} | class : {output_size}")



y_train = np.argmax(y_train, axis = 1)
y_val = np.argmax(y_val, axis = 1)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_val = torch.Tensor(x_val)
y_val = torch.Tensor(y_val).to(torch.long)

epochs = 1000
batch_size = 100
learning_rate= 0.05

model = torch_MLP(input_size= num_feature, hidden1 = 30, hidden2= 20, hidden3 = 10, output_size = output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum= 0.6)
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

model.train()

for epoch in range(epochs + 1):
    epoch_loss = 0.0

    for iteration in range(total_num // batch_size):
        seed = np.random.choice(total_num, batch_size)

        x_batch = torch.Tensor(x_train[seed])
        y_batch = torch.Tensor(y_train[seed]).to(torch.long)
        #print(y_batch.shape)
        y_pred = model(x_batch)
        #print(y_pred.shape)
        optimizer.zero_grad()

        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / batch_size
    #train_output = model(x_train)
    #train_loss = criterion(train_output, y_train)

    #train_output = torch.argmax(train_output, 1)
    
    #train_score = len(train_output[train_output == y_train]) / len(y_train)

    val_output = model(x_val)
    val_loss = criterion(val_output, y_val)

    val_output = torch.argmax(val_output, 1)

    val_score = len(val_output[val_output == y_val])/ len(y_val)
    
    if epoch % 100 == 0:
        print(f"epoch : {epoch} | trian_loss : {epoch_loss} | val_loss : {val_loss} | val_score : {val_score}")