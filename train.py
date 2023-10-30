import numpy as np
from initialize_weights import initialize_weights
from forward_propagation import forward_propagation
from backward_propagation import backward_propagation
from sse import calculate_sse_loss,calculate_mse_loss

def train(X, y, hidden_dim, output_dim, epochs, learning_rate, momentum_rate):
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    hidden_weights, output_weights, hidden_bias, output_bias = initialize_weights(input_dim, hidden_dim, output_dim)
    
    # Initialize momentum-related variables
    prev_hidden_weight_delta = np.zeros_like(hidden_weights)
    prev_output_weight_delta = np.zeros_like(output_weights)
    prev_hidden_bias_delta = np.zeros_like(hidden_bias)
    prev_output_bias_delta = np.zeros_like(output_bias)
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X)):
            inputs = X[i]
            targets = y[i]
            hidden_outputs, output_outputs = forward_propagation(inputs, hidden_weights, output_weights, hidden_bias, output_bias)
            
            prev_hidden_weight_delta, prev_output_weight_delta, prev_hidden_bias_delta, prev_output_bias_delta = backward_propagation(inputs, targets, hidden_outputs, output_outputs, hidden_weights, output_weights, hidden_bias, output_bias, learning_rate, momentum_rate, prev_hidden_weight_delta, prev_output_weight_delta, prev_hidden_bias_delta, prev_output_bias_delta)
            
            loss = calculate_mse_loss(targets, output_outputs)
            total_loss += loss
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")
            
    return hidden_weights, output_weights, hidden_bias, output_bias
