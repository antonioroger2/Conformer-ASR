import torch
import torch.nn as nn

def count_parameters(model):
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename):
    """Saves the model and optimizer states to a file."""
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """
    Loads model, optimizer, and scheduler states from a checkpoint file.
    Returns the last epoch and best loss recorded.
    """
    # Use map_location='cpu' to load on any device initially
    checkpoint = torch.load(filename, map_location='cpu')
    
    # Load model state
    # Handle DataParallel wrapping if present
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Return metadata
    return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))