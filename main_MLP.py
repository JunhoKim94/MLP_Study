import numpy as np
from model.MLP import MLP
from optim.optimizer import SGD, RMSProp, Momentum, Adam
from utils import load_fashion_mnist

np.random.seed(19941017)

x_train, y_train, x_test, y_test = load_fashion_mnist('./data')


epochs = 1000
batch_size = 100
learning_rate= 0.001

num_feature = x_train.shape[1]
total_num = len(x_train)
val_num = len(x_test)
output_size = y_train.shape[1]
print(f"feature number : {num_feature} | train_ data_number : {total_num} | validation_data_number : {val_num} | class : {output_size}")

hidden = [30,20,10]

#optimizer = SGD()
#optimizer = Momentum(0.6)
#optimizer = Adam()
optimizer = RMSProp(0.6, 1e-6)
model = MLP(optimizer = optimizer, learning_rate= learning_rate, input_size= num_feature, output_size = output_size, hidden = hidden)

for epoch in range(epochs + 1):
    epoch_loss = 0.0
    for iteration in range(total_num // batch_size):
        seed = np.random.choice(total_num, batch_size)
        x_batch = x_train[seed]
        y_batch = y_train[seed]

        loss = model.loss(x_batch, y_batch)
        epoch_loss += loss

        model.backward()
        model.update()

    epoch_loss /= batch_size

    #pred = model.predict(x_train)
    #target = np.argmax(y_train, axis = 1)

    #score = len(np.where(pred == target)[0]) / len(target)
    
    val_pred = model.predict(x_test)
    val_loss = model.loss(x_test, y_test)
    val_target = np.argmax(y_test, axis = 1)

    val_score = len(np.where(val_pred == val_target)[0]) / len(val_target)

    if (epoch % 100) == 0:
        print(f"epoch : {epoch} | loss : {epoch_loss} | val_loss : {val_loss} | validation score : {val_score}")
