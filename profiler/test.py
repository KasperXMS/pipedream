import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18

model = resnet18()
symbolic_traced : torch.fx.GraphModule = torch.fx.symbolic_traced(model)
print(symbolic_traced.graph)