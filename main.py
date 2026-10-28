import torch
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
  def __init__(self):
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28*28, 512),
      nn.ReLu(),
      nn.Linear(512, 512),
      nn.ReLu(),
      nn.Linear(512, 10),
    )
    def forward(self, x):
      x = self.flatten(x)
      logits = self.linear_relu_stack(x)
      return logits

model = NeuralNetwork().to(device)
print(model)
