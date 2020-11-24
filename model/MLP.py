from model.layers import Init_Layer, Hidden_layer, SoftmaxwithLoss
from model.Activation import Sigmoid, ReLU
import numpy as np
import copy
import pickle

class MLP:
    def __init__(self, input_size, optimizer, output_size,  learning_rate, hidden, initialize = "xavier"):

        self.layers = [
                        Init_Layer(input_size , hidden[0], ReLU() , initialize),
                        #Hidden_layer(hidden[0], hidden[1], ReLU()),
                        #Hidden_layer(hidden[1], hidden[2], ReLU())
                        ]
        #[10,10,10,10]
        
        for i in range(len(hidden) - 1):
            h = Hidden_layer(hidden[i], hidden[i+1], ReLU(), initialize)
            self.layers.append(h)
        
        self.output_layer = SoftmaxwithLoss(hidden[-1], output_size, initialize)
        self.optimizer = []

        for i in range(len(self.layers) + 2):
            temp = copy.deepcopy(optimizer)
            self.optimizer.append(temp)

        #print(self.optimizer)
        self.learning_rate = learning_rate

    def zero_grad(self):
        self.output_layer.dW = None
        self.output_layer.db = None

        for layer in self.layers:
            layer.dW = None
            layer.db = None


    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        
        y_pred = self.output_layer.predict(x)
        pred = np.argmax(y_pred, axis = -1)

        return pred

    def loss(self, x, y):
        for layer in self.layers:
            x = layer.forward(x)
        
        loss = self.output_layer.forward(x,y)

        return loss

    def backward(self):
        d_prev = 1
        d_prev = self.output_layer.backward(d_prev)
        for layer in self.layers[::-1]:
            d_prev = layer.backward(d_prev)

    def update(self):
        
        self.output_layer.W = self.optimizer[-1].update(self.output_layer.W, self.output_layer.dW, self.learning_rate)
        self.output_layer.b = self.optimizer[-1].update(self.output_layer.b, self.output_layer.db, self.learning_rate)

        for i , layer in enumerate(self.layers[:-1]):
            layer.W = self.optimizer[i].update(layer.W, layer.dW, self.learning_rate)
            layer.b = self.optimizer[i].update(layer.b, layer.db, self.learning_rate)

    def save_weights(self, save_path):
        weights = []
        for layer in self.layers:
            weights.append([layer.W, layer.b])
        
        save_dict = {"weights" : weights}
        with open(save_path, "wb") as f:
            pickle.dump(save_dict, f)
    
    def load_weights(self, save_path):
        with open(save_path, "rb") as f:
            a = pickle.load(f)
        
        weights = a["weights"]

        for idx, layer in enumerate(self.layers):
            layer.W = weights[idx][0]
            layer.b = weights[idx][1]