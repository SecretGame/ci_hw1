import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def conf_mattrix(actual,predicted):
    # Flatten the nested lists
    actual = [item.tolist() for item in actual]
    # print(actual)
    # print(predicted)
    actual_flat = [1 if item == [1, 0] else 0 for item in actual]
    predicted_flat = [1 if item == [1, 0] else 0 for item in predicted]
    # print(actual_flat)
    # print(predicted_flat)

    # Create the confusion matrix
    confusion = confusion_matrix(actual_flat, predicted_flat)
    # print(confusion)
    TP = confusion[1][1]
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['[0,1]', '[1,0]'], 
                yticklabels=['[0,1]', '[1,0]'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.show()
    return accuracy
