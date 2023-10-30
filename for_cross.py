import numpy as np


def output_cross(inputs):
    output = []
    for i in range(len(inputs)):
        a = inputs[i][0]
        b = inputs[i][1]
        if a>b:
            a = 1
            b = 0
        else:
            a = 0
            b = 1
        output = [a,b]
    return output

# inputs = np.array([[0.9,0.8]])
# print(output_cross(inputs))