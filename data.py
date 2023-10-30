import numpy as np
def getdataFile():
    with open('C:\\Users\\Admin\\Desktop\\FloodData.txt', 'r') as file:
        text = file.read()

    # Split the text into lines
    lines = text.strip().split('\n')

    # Initialize lists to store the data
    data = []
    predict = []

    # Parse the text and populate the lists
    for line in lines[1:]:  # Skip the header line
        values = line.split()
        data_row = [int(value)/1000 for value in values[:8]]
        t_plus_7_row = [int(values[8])/1000]  # Convert the t+7 value into a list
        data.append(data_row)
        predict.append(t_plus_7_row)

    # Print the resulting lists
    # print("Data:")
    # print(data)

    # print("\nT+7:")
    # print(predict)

    return np.array(data),np.array(predict)



