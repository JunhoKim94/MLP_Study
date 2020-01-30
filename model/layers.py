import numpy as np

class Init_Layer:
    def __init__(self, input_size, hidden_size, activation, initialize = "xavier"):
        #W : input (D X H), b = H
        if initialize.lower() == "xavier":
            n = (6/(input_size + hidden_size))**0.5
        elif initialize.lower() == "kaiming":
            n = (6 / input_size)**0.5
        else:
            n = 1
        self.W = np.random.uniform(low = -n, high = n, size = (input_size, hidden_size))        
        self.b = np.zeros(hidden_size)
        self.dW , self.db = None, None
        self.activation = activation

    def forward(self, x):
        '''
        input: x (N,D)
        output: h (N,H)
        '''
        output = None
        self.x = x
        
        output = np.matmul(self.x,self.W) + self.b
        #print(output)
        output = self.activation.forward(output)

        return output

    def backward(self, d_prev):
        '''
        input: d_prev (N,H)
        x : (N,D)
        output: dW : (D,H)
                db : (H,)
        '''
        #l : N,H
        l = self.activation.backward(d_prev)
        self.dW = np.matmul(self.x.T,l)
        self.db = np.sum(l, axis = 0)

class Hidden_layer:
    def __init__(self, input_size, hidden_size, activation, initialize = "xavier"):
        #W : input (D X H), b = H
        if initialize.lower() == "xavier":
            n = (6/(input_size + hidden_size))**0.5
        elif initialize.lower() == "kaiming":
            n = (6 / input_size)**0.5
        else:
            n = 1
        self.W = np.random.uniform(low = -n, high = n, size = (input_size, hidden_size))        
        self.b = np.zeros(hidden_size)
        self.dW , self.db = None, None
        self.activation = activation

    def forward(self, x):
        '''
        input: x (N,D)
        output: h (N,H)
        '''
        output = None
        self.x = x
        
        output = np.matmul(self.x,self.W) + self.b
        output = self.activation.forward(output)

        return output

    def backward(self, d_prev):
        '''
        input: d_prev (N,H)
        x : (N,D)
        output: dW : (D,H)
                db : (H,)
        '''
        #l : N,H
        l = self.activation.backward(d_prev)
        self.dW = np.matmul(self.x.T,l)
        self.db = np.sum(l, axis = 0)

        #to update front hidden layer
        dx = np.matmul(l,self.W.T)

        return dx

class SoftmaxwithLoss:
    def __init__(self, input_size, output_size, initialize = "xavier"):
        #inputsize = D, outputsize = O
        if initialize.lower() == "xavier":
            n = (6/(input_size + hidden_size))**0.5
        elif initialize.lower() == "kaiming":
            n = (6 / input_size)**0.5
        else:
            n = 1
        self.W = np.random.uniform(low = -n, high = n, size = (input_size, output_size))        
        self.b = np.zeros(output_size)        
        self.dW, self.db = None, None
        self.x , self.y = None,None
        self.y_pred = None
        self.loss = None

    def forward(self, x, y):
        '''
        input: x = value from last hidden layer (N, D)
               y = target vector (N,O)
        calculate loss
        '''

        
        self.x = x
        self.y = y

        self.y_pred = self.predict(self.x)
        self.loss = self.ce_loss(self.y_pred, self.y)

        return self.loss


    def backward(self, d_prev):
        '''
        d_prev : output N,O
        
        Hidden ---- activation ----  Output
        '''
        batch_size = self.y.shape[0]
        grad = (self.y_pred - self.y)/batch_size
        
        self.dW = np.matmul(self.x.T, grad)
        self.db = np.sum(grad, axis = 0)

        dx = np.matmul(grad, self.W.T)

        return dx
        
    def softmax(self, z):
        #numerically stable softmax
        z = z - np.max(z, axis =1 , keepdims= True)
        _exp = np.exp(z)
        _sum = np.sum(_exp,axis = 1, keepdims= True)
        sm = _exp / _sum

        return sm

    def ce_loss(self, y_pred, y_t):
        '''
        input
        y_pred : prediction (N,O)
        y_t : label (N,O)

        output
        loss: cross-entropy loss
        sigma (y_pred x ln(y_t)) / N
        '''
        eps = 1e-10
        
        loss = -np.sum(y_t * np.log(y_pred + eps)) / len(y_pred)
        
        return loss

    def predict(self, x):

        output = np.matmul(x, self.W) + self.b
        output = self.softmax(output)

        return output