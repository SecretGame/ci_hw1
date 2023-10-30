import numpy as np
from crosspad_data import getCross_data
from sse import calculate_mse_loss, calculate_sse_loss
from train import train
from forward_propagation import forward_propagation
from data import getdataFile

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])

    # X , y = getdataFile()
    X , y = getCross_data()
    input_dim = X.shape[1]
    hidden_dim = 6
    output_dim = y.shape[1]
    epochs = 10000
    learning_rate = 0.4
    momentum_rate = 0.0009

    #Cross-validation
    num_splits = 10
    data_length = len(X)
    segment_length = data_length // num_splits
    print(f"segment_length before: {segment_length}")

    
    validation_losses = []
    for i in range(num_splits):
        print(f"----------------------------------------------{i}")
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length

        if i == num_splits - 1:
            end_idx = data_length  # Ensure that the last segment includes all remaining data points
        
        val_indices = np.arange(start_idx, end_idx)
        train_indices = np.setdiff1d(np.arange(data_length), val_indices)
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
    
        # Train on the training set
        hidden_weights, output_weights, hidden_bias, output_bias = train(X, y, hidden_dim, output_dim, epochs, learning_rate, momentum_rate)
        
        #Validate model
        total_loss = 0
        for j in range(len(X_val)):
            inputs = X_val[j]
            targets = y_val[j]
            hidden_outputs, output_outputs = forward_propagation(inputs, hidden_weights, output_weights, hidden_bias, output_bias)
            print(f"Input: {inputs}, Predicted Output: {output_outputs}")
            loss = calculate_mse_loss(targets,output_outputs)
            total_loss += loss
        avg_loss = total_loss / data_length
        print(f"Average Loss: {avg_loss}")
        validation_losses.append(avg_loss)

 # Calculate average loss over all validation sets
    average_validation_loss = np.mean(validation_losses)
    print(f"Average Validation Loss: {average_validation_loss}")