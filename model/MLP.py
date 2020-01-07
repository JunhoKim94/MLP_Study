from model.layers import Init_Layer, Hidden_layer, SoftmaxwithLoss
from model.Activation import Sigmoid, ReLU
import numpy as np


class MLP:
    def __init__(self, input_size, optimizer, output_size,  learning_rate):

        self.layers = [
                        Init_Layer(input_size ,10, ReLU()),
                        Hidden_layer(10, 10, ReLU()),
                        Hidden_layer(10, 10, ReLU())
                        ]

        self.output_layer = SoftmaxwithLoss(10, output_size)
        self.optimizer = optimizer
        self.learning_rate = learning_rate

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
        
        self.output_layer.W = self.optimizer.update(self.output_layer.W, self.output_layer.dW, self.learning_rate)
        self.output_layer.b = self.optimizer.update(self.output_layer.b, self.output_layer.db, self.learning_rate)

        for layer in self.layers:
            layer.W = self.optimizer.update(layer.W, layer.dW, self.learning_rate)
            layer.b = self.optimizer.update(layer.b, layer.db, self.learning_rate)
