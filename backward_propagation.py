import numpy as np
from activation_functions import sigmoid_derivative
from activation_functions import leaky_relu_derivative

def backward_propagation(inputs, targets, hidden_outputs, output_outputs, hidden_weights, output_weights,
                          hidden_bias, output_bias, learning_rate, momentum_rate, prev_hidden_weight_delta, prev_output_weight_delta,
                            prev_hidden_bias_delta, prev_output_bias_delta):
    output_error = targets - output_outputs
    output_local_variable = output_error * sigmoid_derivative(output_outputs)
    # output_local_variable = output_error * leaky_relu_derivative(output_outputs)
    hidden_error = output_local_variable.dot(output_weights.T)
    hidden_local_variable = hidden_error * sigmoid_derivative(hidden_outputs)
    # hidden_local_variable = hidden_error * leaky_relu_derivative(hidden_outputs)

    output_weight_delta = hidden_outputs.T.dot(output_local_variable) * learning_rate + momentum_rate * prev_output_weight_delta
    hidden_weight_delta = np.dot(inputs.reshape(-1, 1), hidden_local_variable.reshape(1, -1)) * learning_rate + momentum_rate * prev_hidden_weight_delta
    
    output_bias_delta = np.sum(output_local_variable, axis=0) * learning_rate + momentum_rate * prev_output_bias_delta
    hidden_bias_delta = np.sum(hidden_local_variable, axis=0) * learning_rate + momentum_rate * prev_hidden_bias_delta
    
    # อัพเดท weights และ biases
    output_weights += output_weight_delta
    hidden_weights += hidden_weight_delta
    
    output_bias += output_bias_delta
    hidden_bias += hidden_bias_delta

    # อัพเดทข้อมูลที่ t-1
    prev_output_weight_delta = output_weight_delta
    prev_hidden_weight_delta = hidden_weight_delta
    prev_output_bias_delta = output_bias_delta
    prev_hidden_bias_delta = hidden_bias_delta
    
    return hidden_weight_delta, output_weight_delta, hidden_bias_delta, output_bias_delta