from typing import List

from torch import nn


class Pipeline(object):
    def __init__(self, modules: List) -> None:                
        for module in modules:
            if not issubclass(module, nn.Module):
                raise TypeError(f"module must be a subclass of nn.Module, but is not")
        self.modules = nn.ModuleList(modules)
    
    def forward(self, x):
        for module in self.modules:
            x = module.forward(x)
        return x
        
    def train(self, **kwargs):
        for module in self.modules:
            module.train(**kwargs)
