import torch
import torch.nn as nn


class MIMOModel(nn.Module):
    def __init__(self, n_inputs:int, base_model:nn.Module) -> None:
        super().__init__()
        self.n_inputs = n_inputs
        self.base_model = base_model
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if(x.shape[1] == 1): # Assume that x has the shape batch_size x n_inputs x (normal input shape) or batch_size x 1 x (normal input shape)
            x = x.expand(*[-1  if i != 1 else self.n_inputs for i in range(x.ndim)])
        elif(x.shape[1] != self.n_inputs):
            raise AssertionError(f"The model requires {self.n_inputs} different inputs for the subnetworks during training, not {x.shape[1]}.")
        
        outputs:torch.Tensor = self.base_model(torch.flatten(x, start_dim=1, end_dim=2)) # Shape : batch_size x output_shape * n_inputs
        return outputs.view(x.shape[0], self.n_inputs, -1) # Shape : batch_size x n_inputs x output_shape
            