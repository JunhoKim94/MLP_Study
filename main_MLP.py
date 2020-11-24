import numpy as np
from model.MLP import MLP
from optim.optimizer import SGD, RMSProp, Momentum, Adam
from utils import load_fashion_mnist
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = load_fashion_mnist('./data')
x_train, x_val , y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

epochs = 60
batch_size = 32
learning_rate= 0.001
verbose = 5
weight_decay = 1e-3

num_feature = x_train.shape[1]
total_num = len(x_train)
val_num = len(x_val)
test_num = len(x_test)
output_size = y_train.shape[1]

print(f"feature number : {num_feature} | train_data_number : {total_num} | validation_data_number : {val_num} | test_data_number : {test_num} | class : {output_size}")

hidden = [80,40]

#optimizer = SGD()
#optimizer = Momentum(0.6)
optimizer = Adam(beta_1 = 0.9, beta_2 = 0.999, weight_decay = weight_decay)
#optimizer = RMSProp(0.6, 1e-6)
model = MLP(optimizer = optimizer, 
            learning_rate= learning_rate, 
            input_size= num_feature, 
            output_size = output_size, 
            hidden = hidden, 
            initialize= 'kaiming')


acc_stack = []
loss_stack = []
val_acc_stack = []

st = time.time()
best_score = 0
for epoch in range(epochs + 1):
    epoch_loss = 0.0
    for iteration in range(total_num // batch_size):
        seed = np.random.choice(total_num, batch_size)
        x_batch = x_train[seed]
        y_batch = y_train[seed]

        loss = model.loss(x_batch, y_batch)
        epoch_loss += loss

        model.zero_grad()
        model.backward()
        model.update()

    epoch_loss /= batch_size

    pred = model.predict(x_train)
    target = np.argmax(y_train, axis = 1)
    score = len(np.where(pred == target)[0]) / len(target)
    
    val_pred = model.predict(x_val)
    val_target = np.argmax(y_val, axis = 1)
    val_score = len(np.where(val_pred == val_target)[0]) / len(val_target)

    acc_stack.append(score)
    val_acc_stack.append(val_score)
    loss_stack.append(epoch_loss)

    if (epoch % verbose) == 0:
        spend = round((time.time() - st), 3)
        print(f"epoch : {epoch} | loss : {epoch_loss} | train_score : {score} | validation score : {val_score} | time : {spend} seconds")

    if best_score < val_score:
        best_score = val_score
        model.save_weights("./best.pickle")

print(f"total_time : {time.time() - st}")

model.load_weights("./best.pickle")
test_pred = model.predict(x_test)
test_target = np.argmax(y_test, axis = 1)
test_score = len(np.where(val_pred == val_target)[0]) / len(val_target)

print(test_score)

def plot_graph(loss_stack, acc_stack, val_acc_stack):
    a = [i for i in range(epochs + 1)]
    
    #plt.figure(figsize = (10,8))
    fig , ax1 = plt.subplots()
    ax2 = ax1.twinx()
    acc = ax1.plot(a, acc_stack, 'r', label = 'Accuracy')
    loss = ax2.plot(a, loss_stack, 'b', label = 'loss')
    val_acc = ax1.plot(a, val_acc_stack, "g", label = "Val Accuracy")
    plt.legend()
    ax1.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax1.set_ylabel("accuracy")

    ax = acc + loss + val_acc
    labels = [l.get_label() for l in ax]
    plt.legend(ax, labels, loc =2)

    plt.show()

plot_graph(loss_stack, acc_stack, val_acc_stack)