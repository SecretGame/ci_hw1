import numpy as np

def calculate_sse_loss(targets, output_outputs):
    return 0.5 * np.sum((targets - output_outputs) ** 2)

def calculate_mse_loss(targets, output_outputs):
    return np.mean((targets - output_outputs) ** 2)