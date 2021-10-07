import sys
import pandas as pd
import Class
import numpy as np

output_of_fp = []
all_next_input = []
error_array = []

# The Forward Propagation
def fp(input_m, weight, hidden_layer, hidden_layer_arr, output_size, output_layer_arr):
    global output_of_fp, all_next_input
    for i in range(len(hidden_layer) + 1):
        if i == 0:
            # Input and Hidden layer 1
            all_next_input, next_input = Class.Propagation.calculate_Output(all_next_input, weight[i], hidden_layer[i], hidden_layer_arr, input_m, 0)

        elif i == len(hidden_layer):
            # Hidden layer with the Output
            output_of_fp = Class.Propagation.calculate_Output(all_next_input, weight[i], output_size, output_layer_arr, next_input, -1)

        else:
            # Hidden layer i with Hidden layer i+1
            all_next_input, next_input = Class.Propagation.calculate_Output(all_next_input, weight[i], hidden_layer[i], hidden_layer_arr, next_input, i)

# The Backward Propagation
def bp(x_input,y,output_of_fp, hidden_layer, hidden_layer_arr,weight_arr, output_size, all_next_input, learning_rate, output_layer_arr, input_size):
    global error_array
    for i in range(len(hidden_layer) + 1):
        if i == 0:
            # Hidden layer with the Output

            # Error Calculation
            error_array = Class.Propagation.all_output_error_calculation(error_array, output_of_fp, y)

            weight_for_the_layer = weight_arr[len(weight_arr) - 1]
            outputs = all_next_input[len(all_next_input) - 1]

            # Update weight
            Class.Propagation.update_weight(hidden_layer[len(hidden_layer) - 1], output_size, len(weight_arr)-1,
                          weight_for_the_layer, learning_rate, error_array, outputs, weight_arr)

            # Update error
            Class.Propagation.update_error(hidden_layer[len(hidden_layer) -1], output_size, len(all_next_input) - 1,
                         weight_for_the_layer, outputs, error_array, all_next_input)

            # Update bias
            Class.Propagation.update_bias(len(output_layer_arr), output_layer_arr, learning_rate, error_array)
        elif i == len(hidden_layer):
            # Input and Hidden layer 1

            weight_for_the_layer = weight_arr[0]
            output_of_h1 = all_next_input[0]

            # Update weight
            Class.Propagation.update_weight(input_size, hidden_layer[len(hidden_layer) - i], 0,
                          weight_for_the_layer, learning_rate, output_of_h1, x_input, weight_arr)

            # Update bias
            Class.Propagation.update_bias(hidden_layer[len(hidden_layer) - i], hidden_layer_arr[0], learning_rate, output_of_h1)
        else:
            # Hidden layer i with Hidden layer i + 1

            weight_for_the_layer = weight_arr[len(weight_arr) - (i +1)]
            this_hidden_layer = all_next_input[len(all_next_input) - i]
            other_hidden_layer = all_next_input[len(all_next_input) - i - 1 ]

            # Update weight
            Class.Propagation.update_weight(hidden_layer[len(hidden_layer) - (i + 1)], hidden_layer[len(hidden_layer) - i], len(weight_arr) - i - 1,
                          weight_for_the_layer, learning_rate, this_hidden_layer, other_hidden_layer, weight_arr)

            # Update error
            Class.Propagation.update_error(hidden_layer[len(hidden_layer) - i- 1], hidden_layer[len(hidden_layer) - i], len(all_next_input) - i - 1,
                         weight_for_the_layer, other_hidden_layer, this_hidden_layer, all_next_input)

            # Update bias
            Class.Propagation.update_bias(len(this_hidden_layer), hidden_layer_arr[len(hidden_layer_arr) - i], learning_rate, this_hidden_layer)



if __name__ == "__main__":
    NumOfParams = len(sys.argv)
    print("Number of Parameter i s: ", NumOfParams)

    Class.Propagation.set_default_value(0)

    Class.Propagation.read_switch(NumOfParams)

    # Do not give space while giving the values
    # Example of an input given below -->
    # py Backpropagation.py -d Test.csv -h [3] -iw [[-0.5,0.25,0.5,-0.25,-0.12,0.3,0.4,0.2,-0.1],[0.3,-0.2,0.15]] -ib [[0.19,-0.22,0.45],[-0.01]] -lr 0.04 -e 1 -ol y

    print("Dataset Name is   (-b) : ",Class.Propagation.datasetName)
    print("Hidden Layer is   (-h) : ",Class.Propagation.hidden_layer)
    print("Weight is         (-iw): ",Class.Propagation.weight)
    print("Bias is           (-ib): ",Class.Propagation.bias)
    print("Learning Rate is  (-lr): ",Class.Propagation.learning_rate)
    print("Epoch is          (-ep): ",Class.Propagation.epoch)
    print("OutputLevel is    (-ol): ",Class.Propagation.outputLevel)

    Class.Propagation.read_dataset(Class.Propagation.datasetName)

    dataset_x_with_encoding = Class.Propagation.encoding_dataset(Class.Propagation.x)

    Class.Propagation.x = dataset_x_with_encoding

    dataset_y_with_encoding = Class.Propagation.encoding_dataset(Class.Propagation.y)

    Class.Propagation.y = dataset_y_with_encoding

    print("\nDataset before normalization")
    print("\n",Class.Propagation.x)
    print("\n",Class.Propagation.y)

    dataset_x_with_Normalize = Class.Propagation.normalize_dataset(Class.Propagation.x)

    Class.Propagation.x = dataset_x_with_Normalize

    print("\nDataset after normalization")
    print("\n", pd.DataFrame(Class.Propagation.x))
    print("\n", pd.DataFrame(Class.Propagation.y))

    x = Class.Propagation.x
    y = Class.Propagation.y

    input_size = np.shape(x)[1]
    output_size = len(y.target.unique())
    y_unique = y.target.unique()

    print("\n\n")
    # Hidden layer
    hidden_layer = Class.Propagation.read_hidden_layer(Class.Propagation.hidden_layer)
    print("Hidden Layer", hidden_layer)

    # Weight
    weight = Class.Propagation.read_weight(input_size, hidden_layer,output_size,Class.Propagation.weight)
    print("Weights ", weight)

    # Bias
    bias = Class.Propagation.read_bias(hidden_layer,output_size,Class.Propagation.bias)
    print("Bias ", bias)

    x = x.T
    y =y.to_numpy()
    y = y.T

    # This function calculates the weight requirement
    all_sum = Class.Propagation.weight_required(hidden_layer, input_size, output_size)

    # This function separate the weight based on the layers
    weight_arr = Class.Propagation.weight_separator(all_sum, weight)

    # This function separate the hidden layer and output layer bias based on the layers
    hidden_layer_arr, output_layer_arr = Class.Propagation.hidden_layer_and_output_layer_bias_separator(hidden_layer, bias)

    print("\n\n")
    print("Weight Array")
    print(weight_arr)

    print("Hidden Layer Array")
    print(hidden_layer_arr)

    print("Output Layer Array")
    print(output_layer_arr)

    lr = float(Class.Propagation.learning_rate)
    ep = int(Class.Propagation.epoch)

    y = y.flatten()
    x_wiith_size = len(Class.Propagation.x)
    print("\n\n")


    for i in range(ep):
        for i in range(x_wiith_size):
            # Forward Propagation
            all_next_input = []
            fp(Class.Propagation.x[i], weight_arr, hidden_layer, hidden_layer_arr, output_size, output_layer_arr)

            # Backward Propagation
            bp(Class.Propagation.x[i], y[i], output_of_fp, hidden_layer, hidden_layer_arr, weight_arr, output_size,
               all_next_input, lr, output_layer_arr, input_size)

    print("\nUpdated Weight Array\n")
    print(weight_arr)

    print("\nUpdated Hidden layer Array (Bias) \n")
    print(hidden_layer_arr)

    print("\nUpdated Output layer Array (Bias) \n")
    print(output_layer_arr)

    print("\nUpdated Hidden layer Error\n")
    for i in range(len(all_next_input)- len(hidden_layer), len(all_next_input)):
        print(all_next_input[i])

    print("\nUpdated Output layer Error\n")
    for i in range(len(error_array) - len(y_unique), len(error_array)):
        print(error_array[i])
    print("\n")

