import torch
import torch.nn as nn
from pyitlib import discrete_random_variable as drv
from dynib.estimators.TishbyEstimator import TishbyEstimator
from dynib.estimators.NpeetEstimator import NpeetEstimator
from dynib.estimators.TorchEstimator import TorchEstimator

class IBHook:
    def __init__(self, model, estimator="npeet", sigma=0.4, num_bins=256):
        model.hook = self
        #self.estimator = TorchEstimator(normalize=False, device="cuda" if torch.cuda.is_available() else "cpu", num_bins=num_bins, sigma=sigma)
        
        if estimator == "tishby":
            self.estimator = TishbyEstimator()
        elif estimator == "npeet":
            self.estimator = NpeetEstimator(normalize=True)
        
        self.register_all(model)
        self.clear()

    def record_input(self, X, y, onehot=True):
        self.input = X.detach().cpu().numpy()
        yh = y.clone()
        if onehot:
            yh = torch.nn.functional.one_hot(y)
        self.output = yh.detach().cpu().numpy()

    def clear(self):
        self.HY = dict()
        self.XH = dict()
        self.HH = dict()
        self.HH_flat = dict()
        self.input = None 
        self.output = None
        self.input_activations = dict()
        self.output_activations = dict()

    def register_all(self, model):
        self.layers = []
        for i, module in enumerate(model.modules()):
            if module.__class__.__name__ == model.__class__.__name__:
                continue
            module._order = "%s.%d" % (module.__class__.__name__, i)
            module.register_forward_hook(self.forward_hook)
            self.layers.append(module._order)

    def forward_hook(self, module, input, output):
        self.input_activations[module._order] = input[0]#.detach().cpu().numpy()
        self.output_activations[module._order] = output#.detach().cpu().numpy()
        #self.XH[module._order], self.HY[module._order] = self.estimator.compute(0, self.output_activations[module._order], self.input, self.output)
        #self.XH[module._order] = self.estimator.compute()
        #self.HY[module._order] = #ee.mi(self.output_activations[module._order], self.output)
            
        #print(module._order)
        #print("\t", input[0].shape)
        #print("\t", output.shape)
        pass

    def compute_h2h(self):
        for layer_i in self.layers:
            if layer_i not in self.HH.keys():
                self.HH[layer_i] = dict()
            x = self.output_activations[layer_i]
            for layer_j in self.layers:
                y = self.output_activations[layer_j]
                if layer_j not in self.HH[layer_i].keys():
                    self.HH[layer_i][layer_j] = ee.mi(x, y)
                    self.HH[layer_j][layer_i] = self.HH[layer_i][layer_j]
                    self.HH_flat[layer_i + "_vs_" + layer_j] = self.HH[layer_i][layer_j]
                    self.HH_flat[layer_j + "_vs_" + layer_i] = self.HH[layer_i][layer_j]
    def backward_hook(self, module, input, output):
        pass