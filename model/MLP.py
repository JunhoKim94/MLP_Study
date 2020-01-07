from model.layers import Init_Layer, Hidden_layer, SoftmaxwithLoss
from model.Activation import Sigmoid, ReLU


class MLP:
    def __init__(self, optimizer):

        self.layers = [
                        Init_Layer(100,50, ReLU),
                        Hidden_layer(50, 50, ReLU),
                        Hidden_layer(50, 25, ReLU),
                        SoftmaxwithLoss(25, 10)]

        self.optimizer = optimizer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)