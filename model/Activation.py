import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        #eps = 1e-10
        self.out = 1/(1 + np.exp(-z))

        return self.out

    def backward(self, d_prev):

        dz = None

        #Sigmoid의 gradient
        dz = self.out * (1 - self.out)
        dz = dz * d_prev

        return dz

class ReLU:
    def __init__(self):
        self.zero_mask = None
        self.output = None

    def forward(self, z):
        '''
        input:
        z
        output:
        z > 0 --> z 
        z < 0 --> 0
        '''
        self.zero_mask = z < 0
        self.output = np.copy(z)
        self.output[self.zero_mask] = 0

        return self.output

    def backward(self,d_prev):
        '''
        
        dz <--- (dRelu) <---- d_prev

        self.output > 0 --> dRelu = 1
        self.output < 0 --> dRelu = 0

        '''

        d_prev[self.zero_mask] = 0
        dz = d_prev

        return d_prev