import numpy as np
import math
import copy
from Node import Node


class ID3:
    def __init__(self):
        self.node_list = []
        self.system_entropy = 0.0
        self.total_length = 5
        # true_false = [<=50K,>50K]

    def calculate_system_entropy(self, true_false):
        total_length = true_false[0] + true_false[1]
        if true_false[0] == 0:
            true_part = 0
        else:
            true_probability = true_false[0] / total_length
            true_part = true_probability * math.log2(true_probability)
        if true_false[1] == 1:
            false_part = 0
        else:
            false_probability = true_false[1] / total_length
            false_part = false_probability * math.log2(false_probability)
        self.system_entropy = - true_part - false_part

    def calculate_gain_id3(self, att_entropy):
        if len(self.node_list) == 0:
            return self.system_entropy - att_entropy
        else:
            return self.node_list[-1].entropy - att_entropy

    def calculate_entropy_attribute(self, attribute_array):
        # ['Age', [['Jove', 0, 1], ['Adult', 2,1], ['Gran', 3,3], ['Madur', 1,2]]]
        entropy = 0.0
        for attribute_value in attribute_array[1]:
            total_length = attribute_value[1] + attribute_value[2]
            total_probability = total_length / self.total_length
            if attribute_value[1] == 0:
                true_part = 0
            else:
                true_probability = attribute_value[1] / total_length
                true_part = true_probability * math.log2(true_probability)
            if attribute_value[2] == 0:
                false_part = 0
            else:
                false_probability = attribute_value[2] / total_length
                false_part = false_probability * math.log2(false_probability)
            entropy += total_probability * (- true_part - false_part)
        return entropy

    def calculate_true_false(self, data, attribute):
        unique, counts = np.unique(data[attribute].to_numpy(), return_counts=True)
        true_false = []

        for attribute_value in unique:
            true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K')]
            false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K')]
            true = true_raw.shape[0]
            false = false_raw.shape[0]
            true_false.append([attribute_value, true, false])
        # print(true_false)
        return true_false

    def id3(self, data):
        true_false = [np.count_nonzero(data[:-1].to_numpy() == '<=50K'), np.count_nonzero(data[:-1].to_numpy() == '>50K')]
        self.total_length = data.shape[0]
        self.calculate_system_entropy(true_false)
        # To calculate all the true_false of each attribute
        tf_array = []
        for attribute in data.columns:
            if attribute != 'Income':
                # Example: [['Family', [['SI', 0, 2], ['NO', 2, 1]]], ['Gran', [['SI', 2, 1], ['NO', 3, 2]]]]
                tf_array.append([attribute, self.calculate_true_false(data, attribute)])

        entropy_array = []
        gain_array = []

        # Calculate the Entropy
        for attribute in tf_array:
            entropy_array.append(self.calculate_entropy_attribute(attribute))
        # Calculate the Gain
        for entropy in entropy_array:
            gain_array.append(self.calculate_gain_id3(entropy))

        # Creating the winner node
        winner_index = gain_array.index(max(gain_array))
        winner_attribute, winner_entropy = list(data.columns)[winner_index], entropy_array[winner_index]
        winner_edges = []
        for true_false in tf_array:
            if true_false[0] == winner_attribute:
                winner_edges = true_false[1]
        winner_node = Node(entropy=winner_entropy, attribute=winner_attribute,  edges=winner_edges)
        self.node_list.append(winner_node)
        print(self.node_list[-1].print_node())
