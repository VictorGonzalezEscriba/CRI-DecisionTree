import numpy as np
import math
import copy
from Node import Node


class ID3:
    def __init__(self):
        self.node_list = []
        self.visited_nodes = []
        self.used_attributes = []
        self.system_entropy = 0.0
        self.total_length = 0
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
            return [att_entropy[0], self.system_entropy - att_entropy[1]]
        else:
            return [att_entropy[0], self.node_list[-1].entropy - att_entropy[1]]

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
        return [attribute_array[0], entropy]

    def check_edge(self, node, edge):
        for son in node.sons:
            if son.inner_edge == edge:
                return False
        return True

    def calculate_true_false(self, data=None, attribute=None, node=None, edge=None):
        # We have to do two cases, the root case and the other ones
        true_false = []
        if node is None:
            unique = np.unique(data[attribute].to_numpy())
            for attribute_value in unique:
                true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K')]
                false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K')]
                true_false.append([attribute_value, true_raw.shape[0], false_raw.shape[0], true_raw, false_raw])
        else:
            # To save the attributes
            unique = np.unique(data[attribute].to_numpy())
            for attribute_value in unique:
                true_raw = edge[3].loc[(edge[3][attribute] == attribute_value) & (edge[3]['Income'] == '<=50K') & (edge[3][node.attribute] == edge[0])]
                false_raw = edge[4].loc[(edge[4][attribute] == attribute_value) & (edge[4]['Income'] == '>50K') & (edge[4][node.attribute] == edge[0])]
                true_false.append([attribute_value, true_raw.shape[0], false_raw.shape[0], true_raw, false_raw])
        return true_false

    def chose_winner(self, tf_array, node=None, edge=None):
        # Calculate the entropy of each attribute
        entropy_array, gain_array = [], []
        # Calculate the Entropy
        for attribute in tf_array:
            entropy_array.append(self.calculate_entropy_attribute(attribute))
        # Calculate the Gain
        if len(entropy_array) != 0:
            for entropy in entropy_array:
                gain_array.append(self.calculate_gain_id3(entropy))
                # Creating the winner node
            gains = []
            for gain in gain_array:
                gains.append(gain[1])
            winner_index = gains.index(max(gains))
            winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
            winner_edges = []

            # Search the winner node's edges
            for true_false in tf_array:
                if true_false[0] == winner_attribute:
                    winner_edges = true_false[1]

            if node is None:
                winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, root=True, inner_edge=edge, father_list=[winner_attribute])
            else:
                n_father_list = copy.deepcopy(node.father_list)
                n_father_list.append(winner_attribute)
                winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, inner_edge=edge, root=False, father=node, father_attribute=node.attribute, father_list=n_father_list)
                node.add_son(winner_node)
            self.node_list.append(winner_node)

    def id3(self, data, node):
        contador = 0
        """
        Condiciones de parada:
        No quedan más atributos en una rama
        Todas las hojas son decisiones
        """
        # En profunditat haber visitat totos els atributs o haber arribat a un decisió
        if node is not None and node.father is not None:
            print('\n\n')
            if len(self.node_list) == 0:
                return self.visited_nodes[0].show_tree()

        # To calculate all the true_false of each attribute
        tf_array = []
        if node is None:
            # System calculations
            true_false = [np.count_nonzero(data['Income'].to_numpy() == '<=50K'), np.count_nonzero(data['Income'].to_numpy() == '>50K')]
            self.total_length = data.shape[0]
            self.calculate_system_entropy(true_false)
            for attribute in data.columns:
                if attribute != 'Income':
                    # Example: [['Family', [['SI', 0, 2], ['NO', 2, 1]]], ['Gran', [['SI', 2, 1], ['NO', 3, 2]]]]
                    tf_array.append([attribute, self.calculate_true_false(data=data, attribute=attribute)])
            self.chose_winner(tf_array=tf_array, edge=true_false)
        else:

            self.visited_nodes.append(node)
            node.set_visited()
            for edge in node.edges:
                if self.check_edge(node, edge):
                    tf_array = []
                    for attribute in data.columns:
                        if attribute != 'Income' and attribute not in node.father_list:
                            # Example: [['Family', [['SI', 0, 2], ['NO', 2, 1]]], ['Gran', [['SI', 2, 1], ['NO', 3, 2]]]]
                            tf_array.append([attribute, self.calculate_true_false(data=data, attribute=attribute, node=node, edge=edge)])
                    self.chose_winner(tf_array=tf_array, node=node, edge=edge)

        if len(self.node_list) == 1:
            return self.id3(data, self.node_list[0])
        else:
            for son in node.sons:
                return self.id3(data, son)
