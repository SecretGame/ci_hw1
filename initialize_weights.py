import numpy as np

def initialize_weights(input_dim, hidden_dim, output_dim):
    np.random.seed(1)
    hidden_weights = np.random.uniform(size=(input_dim, hidden_dim))
    output_weights = np.random.uniform(size=(hidden_dim, output_dim))
    hidden_bias = np.random.uniform(size=(1, hidden_dim))
    output_bias = np.random.uniform(size=(1, output_dim))
    return hidden_weights, output_weights, hidden_bias, output_bias