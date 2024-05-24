import torch
from torchvision import models
from torchvision.models.densenet import DenseNet121_Weights

a = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

