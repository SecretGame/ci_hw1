import numpy as np
from activation_functions import sigmoid,leaky_relu

def forward_propagation(inputs, hidden_weights, output_weights, hidden_bias, output_bias):
    hidden_layer_input = np.dot(inputs, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    # hidden_layer_output = leaky_relu(hidden_layer_input)
    # print("hidden_input: ",hidden_layer_input)
    # print("hidden_output: ",hidden_layer_output)

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    output_layer_output = sigmoid(output_layer_input)
    # output_layer_output = leaky_relu(output_layer_input)
    # print("output_layer_input: ",output_layer_input)
    # print("output_layer_output: ",output_layer_output)
    return hidden_layer_output, output_layer_output
