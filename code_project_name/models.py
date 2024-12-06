import torch
import torch.nn as nn


class MIMOModel(nn.Module):
    def __init__(self, n_inputs:int, base_model:nn.Module) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.base_model = base_model
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if(self.training): # Assume that x has the shape batch_size x n_inputs x (normal input shape)
            assert x.shape[1] == self.n_inputs, f"The model requires {self.n_inputs} different inputs for the subnetworks during training, not {x.shape[1]}."
            return self.base_model(torch.flatten(x, start_dim=1, end_dim=2))
        else:
            assert x.shape[1] == 1, f"Validation uses one input for all subnetworks, not {x.shape[1]}."
            return self.base_model(x.squeeze().expand(*[-1  if i != 1 else self.n_inputs for i in range(x.ndim)]))
            