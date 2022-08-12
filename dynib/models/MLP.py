import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_shape, output_size, flatten=False, hidden_layers=None, bias=True, layerpref="fc", act_fun=nn.ReLU, outact_fun=nn.Softmax):
        super().__init__()
        input_test = torch.zeros(input_shape)
        self.flatten = None
        if flatten:
            self.flatten = nn.Flatten(1)
            #nput_test = self.flatten(input_test)
            #input_shape = input_test.shape

        i = 0
        h_i = input_shape[-1]
        self.layer_names = []
        if hidden_layers is None:
            hidden_layers = []
        hidden_layers.append(output_size)
        for i, h_o in enumerate(hidden_layers):
            layer = nn.Linear(h_i, h_o, bias)
            self.layer_names.append(layerpref + str(i))
            setattr(self, self.layer_names[-1], layer)
            self.layer_names.append("act" + str(i))
            if i == len(hidden_layers) - 1:
                act = outact_fun()
            else:
                act = act_fun()
            setattr(self, self.layer_names[-1], act)
            h_i = h_o

    def forward(self, x):
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)
        return x