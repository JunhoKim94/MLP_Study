from torch import nn, tensor
import torch
import torch.nn.functional as F

class torch_MLP(nn.Module):

    def __init__(self, input_size, hidden1, hidden2, hidden3, output_size):
        super(torch_MLP, self).__init__()
        

        self.input = input_size
        self.hidden1 =hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.output_size = output_size

        self.linear = nn.Sequential(
            nn.Linear(self.input,self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, self.hidden3),
            nn.ReLU(),
            nn.Linear(self.hidden3, self.output_size)
        )

    def forward(self, x):
        '''
        input : x (N,D)
        output : out (N,O)
        '''

        output = self.linear(x)
        output = F.softmax(output, dim = 1)

        return output