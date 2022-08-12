import torch 
import torch.nn as nn

class DictModel(nn.Module):
    def __init__(self, dict_layers):
        super().__init__()
        self.layer_names = dict_layers.keys()
        self.dict_layers = dict_layers

        for k, v in dict_layers.items():
            layer 