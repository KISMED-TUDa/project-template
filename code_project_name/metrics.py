import torch

def accuracy(predictions:torch.Tensor, labels:torch.Tensor):
    assert predictions.shape == labels.shape, """Predictions and labels must have the same shape"""
    if(predictions.ndim > 2): # Assume last dimension contains the one hot representation
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1, predictions.shape[-1])

    # Transform one hot representation to class label representation
    if(predictions.ndim > 1):
        predictions = predictions.argmax(dim=-1)
        labels = labels.argmax(dim=-1)
    
    # Calculate accuracy as number of correct predictions divided by the number of predictions
    correct_predictions = predictions.eq(labels).sum()
    return correct_predictions / predictions.shape[0]

def multilabel_accuracy(predictions:torch.Tensor, labels:torch.Tensor):
    assert predictions.ndim == 2 and labels.ndim == 2, """predictions or labels didn't have 2 dimensions. 
    Only 2D tensors of the shape (n_samples, n_classes) with 0 or 1 for each class are supported."""
    
    # Calculate accuracy as number of correct predictions divided by the number of predictions
    correct_predictions = predictions.eq(labels).sum()
    return correct_predictions / predictions.numel()