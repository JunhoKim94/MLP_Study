import numpy as np
from model.MLP import MLP
from optim.optimizer import SGD, RMSProp, Momentum
from utils import load_fashion_mnist



x_train, y_train, x_test, y_test = load_fashion_mnist('./data')


epochs = 2000
batch_size = 100
learning_rate= 0.05

num_feature = x_train.shape[1]
total_num = len(x_train)
output_size = y_train.shape[1]
print(f"feature number : {num_feature} | data_number : {total_num} | class : {output_size}")

optimizer = SGD(0,0)
model = MLP(optimizer = optimizer, learning_rate= learning_rate, input_size= num_feature, output_size = output_size)

for epoch in range(epochs):
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

    pred = model.predict(x_train)
    target = np.argmax(y_train, axis = 1)

    score = len(np.where(pred == target)[0]) / len(target)
    
    val_pred = model.predict(x_test)
    val_target = np.argmax(y_test, axis = 1)

    val_score = len(np.where(val_pred == val_target)[0]) / len(val_target)

    if (epoch % 10) == 0:
        print(f"epoch : {epoch} | loss : {epoch_loss} | score : {score} | validation score : {val_score}")
