import numpy as np


def getCross_data():
    with open('C:\\Users\\Admin\\Desktop\\cross_1.pat', 'r') as file:
            text = file.read()

    # Split the text into lines
    lines = text.strip().split('\n')

    # Initialize lists for input and output
    input_data = []
    output_data = []

    # Iterate through lines and extract data
    for i in range(0, len(lines), 3):
        if len(lines) > i + 2 and lines[i].startswith("p"):
            input_values = list(map(float, lines[i + 1].split()))
            output_values = list(map(int, lines[i + 2].split()))
            
            input_data.append(input_values)
            output_data.append(output_values)

    # Print the extracted input and output
    # print("Input =", input_data)
    # print("Output =", output_data)

    return np.array(input_data), np.array(output_data)