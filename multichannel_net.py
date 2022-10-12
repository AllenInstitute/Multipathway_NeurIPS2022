import numpy as np
import matplotlib.pyplot as plt
import torch

class MultipathwayNet (torch.nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity=None, bias=False, num_pathways=2, depth=2, width=1000, eps=0.01, hidden=None):   
        
        super(MultipathwayNet, self).__init__()
        
        # hidden is assumed to be a list with entries corresponding to each pathway, each entry a list of the widths of that pathway by depth
        # hidden!=None will override num_pathways, depth, width
        # 'depth' above is the number of weights, not the number of hidden layers

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.nonlinearity = nonlinearity

        self.eps = eps

        if hidden is None:
            hidden = []
            for pi in range(num_pathways):
                pathway = []
                for di in range(depth-1):
                    pathway.append(width)
                hidden.append(pathway)

        self.hidden = hidden

        self.hidden_layers = []

        for pathway in self.hidden:
            op_list = []
            for di, depth in enumerate(pathway):
                if di==0:
                    op = torch.nn.Linear(self.input_dim, depth, bias=self.bias)
                    
                     
                else:
                    op = torch.nn.Linear(pathway[di-1], depth, bias=self.bias)
                  
                op.weight = torch.nn.Parameter(torch.randn(op.weight.shape)*eps, requires_grad=True)
                
                if op.bias is not None:
                    op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
                op_list.append(op)

            op = torch.nn.Linear(pathway[-1], self.output_dim, bias=self.bias)
            op.weight = torch.nn.Parameter(torch.randn(op.weight.shape)*eps, requires_grad=True)
            if op.bias is not None:
                op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
            op_list.append(op)
            

            self.hidden_layers.append(op_list)

        for pi, op_list in enumerate(self.hidden_layers):
            for oi, op in enumerate(op_list):
                self.register_parameter(name= "Path_{}_Depth_{}_weight".format(pi, oi), param=op.weight)
                self.register_parameter(name= "Path_{}_Depth_{}_bias".format(pi, oi), param=op.bias)

        if self.nonlinearity is not None:
            temp_layers = self.hidden_layers
            self.hidden_layers = []
            for op_list in temp_layers:
                new_op_list = []
                for op in op_list:
                    new_op_list.append(op)
                    new_op_list.append(self.nonlinearity)
                self.hidden_layers.append(new_op_list)



    def forward(self, x):

        output = 0

        for op_list in self.hidden_layers:
            xtemp = x
            for op in op_list:
                xtemp = op(xtemp)
            output += xtemp

        return output

    def omega(self):

        with torch.no_grad():
            x = torch.eye(self.input_dim).to(self.hidden_layers[0][0].weight.device)
            output = []

            for op_list in self.hidden_layers:
                xtemp = x
                for op in op_list:
                    xtemp = op(xtemp)
                output.append(xtemp.T.detach())

        return output
