import sys
from sklearn import datasets, preprocessing
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

class Propagation:
    @property
    def datasetName(self):
        return self.datasetName

    @datasetName.setter
    def datasetName(self, datasetName):
        self.datasetName= datasetName


    @property
    def hidden_layer(self):
        return self.hidden_layer

    @hidden_layer.setter
    def hidden_layer(self, hidden_layer):
        self.hidden_layer = hidden_layer


    @property
    def weight(self):
        return self.weight

    @weight.setter
    def weight(self, weight):
        self.weight = weight


    @property
    def bias(self):
        return self.bias

    @bias.setter
    def bias(self, bias):
        self.bias = bias


    @property
    def learning_rate(self):
        return self.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


    @property
    def epoch(self):
        return self.epoch

    @weight.setter
    def epoch(self, epoch):
        self.epoch = epoch

    @property
    def outputLevel(self):
        return self.outputLevel

    @outputLevel.setter
    def outputLevel(self, outputLevel):
        self.outputLevel = outputLevel

    @property
    def x(self):
        return self.x

    @x.setter
    def x(self, x):
        self.x = x

    @property
    def y(self):
        return self.y

    @y.setter
    def y(self, y):
        self.y = y

    # This function sets the default value
    def set_default_value(a):
        Propagation.datasetName = "iris"
        Propagation.hidden_layer = "[2]"
        Propagation.weight = "[[0.2,-0.3,0.4,0.1,-0.5,0.2],[-0.3,-0.2]]"
        Propagation.bias = "[[-0.4,0.2],[0.1]]"
        Propagation.learning_rate = "0.9"
        Propagation.epoch = "1"
        Propagation.outputLevel = ""

    # This function reads the given switch value
    def read_switch(NumOfParams):
        for i in range(1, NumOfParams):
            if sys.argv[i].replace(" ", "") == '-d':
                Propagation.datasetName = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-h':
                Propagation.hidden_layer = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-iw':
                Propagation.weight = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-ib':
                Propagation.bias = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-lr':
                Propagation.learning_rate = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-ep':
                Propagation.epoch = sys.argv[i + 1]
            elif sys.argv[i].replace(" ", "") == '-ol':
                Propagation.outputLevel = sys.argv[i + 1]

    # This function reads the given dataset
    def read_dataset(datasetName):
        if datasetName == "iris":
            data = datasets.load_iris()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "boston":
            data = datasets.load_boston()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "breast_cancer":
            data = datasets.load_breast_cancer()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "diabetes":
            data = datasets.load_diabetes()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "digits":
            data = datasets.load_digits()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "files":
            data = datasets.load_files()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "linnerud":
            data = datasets.load_linnerud()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        elif datasetName == "wine":
            data = datasets.load_wine()
            Propagation.x = pd.DataFrame(data.data, columns=data.feature_names)
            dataset_y = data.target
            Propagation.y = pd.DataFrame(dataset_y, columns=['target'])
        else:
            dataset = pd.read_csv(datasetName)
            Propagation.dataset_target_split(dataset,Propagation.outputLevel)

    # This function dose encoding
    def encoding_dataset(dataset):
        if dataset.shape[0] != 0:
            columns = [column for column in dataset.columns if dataset[column].dtype in ['O']]
            dataset[columns] = dataset[columns].apply(LabelEncoder().fit_transform)
            return dataset
        else:
            return dataset

    # This function dose normalization
    def normalize_dataset(dataset):
        df = preprocessing.normalize(dataset)
        return df

    # This function splits the target
    def dataset_target_split(dataset, classLevel):
        if len(classLevel) != 0:
            y = dataset[classLevel]
            y = y.to_frame()
            y.columns = ["target"]
            Propagation.y = y
            Propagation.x = dataset.drop(classLevel, axis=1)
        else:
            y = dataset.iloc[:, -1]
            y = y.to_frame()
            y.columns = ["target"]
            Propagation.y = y
            Propagation.x = dataset
            Propagation.x = Propagation.x.iloc[:, :-1]

    # This function append the string values in array
    def append_string_in_array(w, z):
        array1 = []
        array2 = []
        count = 0
        for i in range(len(w)):
            if w[i] == "[":
                array1.append(i)
            elif w[i] == "]" and i != len(w) - 1 and z == 0:
                if len(array1) != 0:
                    temp = []
                    for j in range(array1[len(array1) - 1] + 1, i):
                        temp.append(w[j])
                    array2.append(temp)
                    count = count + 1
                    array1.pop(len(array1)-1)
            elif w[i] == "]" and z == 1:
                temp = []
                for j in range(array1[len(array1) - 1] + 1, i):
                    temp.append(w[j])
                array2.append(temp)
                count = count + 1
                array1.pop(len(array1) - 1)
        return array1, array2

    # This function split the values in the array using ","
    def split_values(array2):
        array3 = []
        array4 = []
        for i in range(len(array2)):
            n = array2[i]
            count = - 1
            for j in range(len(n)):
                if n[j] == "," or len(n) == j + 1:
                    array3 = []
                    for k in range(count + 1, j):
                        array3.append(n[k])
                    if len(n) == j + 1:
                        array3.append(n[j])
                    count = j
                    array4.append(array3)
        return array3, array4

    # This function joins the string values and converts them into int or float
    def convert_string_to_float(array4, z):
        weight_val = []
        for i in range(len(array4)):
            temp = array4[i]
            join_string = ''.join(temp)
            if z == 0:
                convert = float(join_string)
            elif z == 1:
                convert = int(join_string)
            weight_val.append(convert)
        return weight_val

    # This function counts the weight requirement
    def count_weight_requirement(hidden_layer_size, input_size, output_size):
        sum = 0
        val = 0
        for i in range(len(hidden_layer_size) + 1):
            if i == 0:
                sum = sum + input_size * hidden_layer_size[i]
            elif i == len(hidden_layer_size):
                sum = sum + output_size * hidden_layer_size[len(hidden_layer_size) - 1]
            else:
                sum = sum + hidden_layer_size[val] * hidden_layer_size[val + 1]
                val = val + 1
        return sum

    # This function reads the given weight using the switch and convert them to array
    # and if the given weight is not enough it assign the weight randomly.
    def read_weight(input_size,hidden_layer_size,output_size,w):
        array1, array2 = Propagation.append_string_in_array(w,0)
        array3, array4 = Propagation.split_values(array2)
        weight_val = Propagation.convert_string_to_float(array4, 0)
        sum = Propagation.count_weight_requirement(hidden_layer_size, input_size, output_size)

        weight_ren = []
        if len(weight_val) != sum:
            print("\n************************************ Warning ************************************\n")
            print("*** The given or default weights are not enough to do the Forward Propagation ***")
            print("***                       So, taking weights randomly.                        ***")
            print("\n************************************ Warning ************************************\n")
            for i in range(sum):
                weight_ren.append(round(random.uniform(-1, 1), 1))
            return weight_ren
        else:
            return weight_val

    # This function reads the given hidden layer using the switch and convert them to array
    def read_hidden_layer(w):
        array1, array2 = Propagation.append_string_in_array(w,1)
        array4, array3 = Propagation.split_values(array2)
        weight_val = Propagation.convert_string_to_float(array3, 1)
        return weight_val

    # This function reads the given bias using the switch and convert them to array
    # and if the given bias is not enough it assign the bias randomly.
    def read_bias(hidden_layer_size, output_size, w):
        array1, array2 = Propagation.append_string_in_array(w, 0)
        array3, array4 = Propagation.split_values(array2)
        weight_val = Propagation.convert_string_to_float(array4, 0)

        sum = 0
        for i in range(len(hidden_layer_size) + 1):
            if i == len(hidden_layer_size):
                sum = sum + output_size
            else:
                sum = sum + hidden_layer_size[i]

        weight_ren = []
        if len(weight_val) != sum:
            print("\n************************************ Warning ************************************\n")
            print("***  The given or default bias are not enough to do the Forward Propagation   ***")
            print("***                       So, taking bias randomly.                           ***")
            print("\n************************************ Warning ************************************\n")
            for i in range(sum):
                weight_ren.append(round(random.uniform(-1, 1), 1))
            return weight_ren
        else:
            return weight_val

    # This function calculates the weight requirement
    def weight_required(hidden_layer, input_size, output_size):
        all_sum = []
        val = 0
        for i in range(len(hidden_layer) + 1):
            sum = 0
            if i == 0:
                sum = input_size * hidden_layer[i]
                all_sum.append(sum)
            elif i == len(hidden_layer):
                sum = output_size * hidden_layer[len(hidden_layer) - 1]
                all_sum.append(sum)
            else:
                sum = hidden_layer[val] * hidden_layer[val + 1]
                all_sum.append(sum)
                val = val + 1
        return all_sum

    # This function separate the weight based on the layers
    def weight_separator(all_sum, weight):
        weight_arr = []
        count = 0
        for i in range(len(all_sum)):
            temp = []
            for j in range(count, count + all_sum[i]):
                if j <= count + all_sum[i]:
                    temp.append(weight[j])
                    count = count + 1
            weight_arr.append(temp)
        return weight_arr

    # This function separate the hidden layer and output layer bias
    def hidden_layer_and_output_layer_bias_separator(hidden_layer, bias):
        hidden_layer_arr = []
        output_layer_arr = []
        count = 0
        for i in range(len(hidden_layer)):
            temp = []
            for j in range(count, count + hidden_layer[i]):
                if j <= count + hidden_layer[i]:
                    temp.append(bias[j])
                    count = count + 1
            hidden_layer_arr.append(temp)
        sum_h = 0
        for i in range(len(hidden_layer)):
            sum_h = hidden_layer[i] + sum_h
        for i in range(1):
            for j in range(sum_h, len(bias)):
                output_layer_arr.append(bias[j])
        return hidden_layer_arr, output_layer_arr

    # This is the activation function
    def activation(input):
        return 1/(1 + np.exp(-input))

    # This function sum the input and bias values
    def input_calculation(bias, output, weight_m):
        sum = 0
        for i in range(len(weight_m)):
            sum += weight_m[i]*output[i]
        return sum + bias

    # This function adjust the weight
    def weight_Adj(old_weight, learning_rate, error, output):
        return old_weight + learning_rate * error * output

    # This function adjust the bias
    def bias_Adj(old_bias , learning_rate, error):
        return old_bias + learning_rate * error

    # This function calculates the output error
    def error_calculation(output, target):
        return output * (1 - output) * (target - output)

    # This function calculates the Backward error
    def error_Backward(output, error_m, weight_m):
        sum = 0
        for i in range(len(weight_m)):
            sum += error_m[i] * weight_m[i]

        return output * (1 - output) * sum

    # This function does all the processes required to get
    # the Forward Propagation output for all hidden layer and output
    def calculate_Output(all_next_input, weights, size, bias_values, input_matrix, index_number):
        next_input_value = []
        weight_for_the_layer = weights
        for j in range(size):
            weight_matrix = []
            for k in range(j, len(weight_for_the_layer), size):
                weight_matrix.append(weight_for_the_layer[k])
            weight_matrix = np.array(weight_matrix)
            if index_number == -1:
                bias_value = bias_values[j]
            else:
                bias_value = bias_values[index_number][j]
            input_calculation_value = Propagation.input_calculation(bias_value, input_matrix, weight_matrix)
            activation_calculation_value= Propagation.activation(input_calculation_value)
            next_input_value.append(activation_calculation_value)
        if index_number == -1:
            return next_input_value
        else:
            all_next_input.append(next_input_value)
            return all_next_input, next_input_value

    # This function calculates the all output error and returns the array (Backward Propagation)
    def all_output_error_calculation(error_array, output_value, value_of_y):
        for j in range(len(output_value)):
            error = Propagation.error_calculation(output_value[j], value_of_y)
            error_array.append(error)
        return error_array

    # This function updates the weight value
    def update_weight(size_j, size_k, index_number, weight_for_the_layer, learning_rate, errors, outputs, weight_arr):
        c = 0
        for j in range(size_j):
            count = 0
            for k in range(c, c + size_k):
                new_weight = Propagation.weight_Adj(weight_for_the_layer[k], learning_rate, errors[count], outputs[j])
                weight_arr[index_number][k] = new_weight
                count = count + 1
            c = c + size_k

    # This function updates the error value
    def update_error(size_j, size_k,index_number, weight_for_the_layer, outputs, error_array, all_next_input):
        new_errors = []
        c = 0
        for j in range(size_j):
            weight_matrix = []
            for k in range(c, c + size_k):
                weight_matrix.append(weight_for_the_layer[k])
            weight_matrix = np.array(weight_matrix)
            new_error = Propagation.error_Backward(outputs[j], error_array, weight_matrix)
            new_errors.append(new_error)
            all_next_input[index_number][j] = new_error
            c = c + size_k

    # This function updates the bias value
    def update_bias(size_j, output_layer_arr, learning_rate, error_array):
        new_bias_value = []
        for j in range(size_j):
            new_bias = Propagation.bias_Adj(output_layer_arr[j], learning_rate, error_array[j])
            new_bias_value.append(new_bias)
            output_layer_arr[j] = new_bias