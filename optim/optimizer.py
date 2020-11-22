import numpy as np


class SGD:
    def __init__(self):
        # ========================= EDIT HERE =========================
        pass


        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        updated_weight = w - lr * grad


        # =============================================================
        return updated_weight

class Momentum:
    def __init__(self, gamma):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        # =============================================================
        self.v = dict()

    def update(self, w, grad, lr):
        updated_weight = None
        if self.v.get(grad.shape) is None:
            self.v[grad.shape] = np.zeros_like(grad)
        
        #print(len(self.v))
        # ========================= EDIT HERE =========================
        #print(len(self.v), grad.shape, self.v)
        '''
        self.v[grad.shape] = self.gamma * self.v[grad.shape] + lr * grad
        updated_weight = w - self.v[grad.shape]

        '''
        self.v[grad.shape] = self.gamma * self.v[grad.shape] + grad
        updated_weight = w - lr * self.v[grad.shape]
        

        # =============================================================
        return updated_weight


class RMSProp:
    # ========================= EDIT HERE =========================
    def __init__(self, gamma, epsilon):
        # ========================= EDIT HERE =========================
        self.gamma = gamma
        self.epsilon = epsilon
        self.G = dict()

        # =============================================================

    def update(self, w, grad, lr):
        updated_weight = None
        # ========================= EDIT HERE =========================
        if self.G.get(grad.shape) is None:
            self.G[grad.shape] = np.zeros_like(grad)

        self.G[grad.shape] = self.gamma*self.G[grad.shape] + (1-self.gamma)*(grad**2)
        #print(self.G.shape,grad.shape)
        updated_weight = w - (lr/np.sqrt(self.epsilon+self.G[grad.shape]))*grad



        # =============================================================
        return updated_weight

class Adam:
    def __init__(self, beta_1 = 0.9, beta_2 = 0.999, weight_decay = 1e-5):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = dict()
        self.v = dict()
        self.iter = 0
        self.weight_decay = weight_decay


    def update(self, w, grad, lr):

        grad = grad + self.weight_decay * w        
        
        if self.v.get(grad.shape) is None or self.m.get(grad.shape) is None:
            self.v[grad.shape] = np.zeros_like(grad)
            self.m[grad.shape] = np.zeros_like(grad)

        self.iter += 1
        lr_t = lr * np.sqrt(1.0 - self.beta_2**self.iter) / (1.0 - self.beta_1 ** self.iter)

        self.m[grad.shape] = self.m[grad.shape] + (1 - self.beta_1) * (grad - self.m[grad.shape])
        self.v[grad.shape] = self.v[grad.shape] + (1 - self.beta_2) * (grad**2 - self.v[grad.shape])

        updated_weight = w - lr_t * self.m[grad.shape] / (np.sqrt(self.v[grad.shape]) + 1e-7)

        return updated_weight

