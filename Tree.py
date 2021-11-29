import numpy as np
import math
import copy
from collections import deque


class Node:
    def __init__(self, father=None, attribute=None, entropy=None, edges=None, inner_edge=None, father_list=None, gini=None, print=None):
        self.attribute = attribute
        self.print = print
        self.father = father
        self.sons = deque()
        self.entropy = entropy
        self.gini = gini
        self.edges = edges
        self.inner_edge = inner_edge
        self.decision = ''
        self.root = False
        self.leaf = False
        self.father_list = father_list
        self.create_leaf()

    def show_tree(self, level=0):
        if not self.root:
            print('\t' * level + '|_', '(' + self.print + ')', ' - ', self.inner_edge.value, '{', self.inner_edge.true, ',', self.inner_edge.false, '}')
        else:
            print('\t' * level + '(' + self.print + ')', ' - ', '{', self.inner_edge.true, ',', self.inner_edge.false, '}')
        for son in self.sons:
            son.show_tree(level+1)

    def create_leaf_son(self, inner_edge, decision):
        son = Node(father=self, inner_edge=inner_edge, attribute='Income', print=decision)
        son.leaf = True
        self.sons.append(son)

    def create_leaf(self):
        if self.edges is not None:
            for edge in self.edges:
                # To avoid {0,0}, we chose the decision with more probability
                if edge.true == 0 and edge.false == 0:
                    true = self.inner_edge.true
                    false = self.inner_edge.false
                    if true > false:
                        self.create_leaf_son(edge, '<=50K')
                    elif true < false:
                        self.create_leaf_son(edge, '>50K')
                    elif true == false:
                        true_f = self.father.inner_edge.true
                        false_f = self.father.inner_edge.false
                        if true_f > false_f:
                            self.create_leaf_son(edge, '<=50K')
                        else:
                            self.create_leaf_son(edge, '>50K')
                else:
                    if edge.true == 0:
                        self.create_leaf_son(edge, '>50K')
                    elif edge.false == 0:
                        self.create_leaf_son(edge, '<=50K')

    def is_decision(self):
        if self.print == '>50K' or self.print == '<=50K':
            return True
        else:
            return False


class Edge:
    def __init__(self, value=None, true=None, false=None, data_true=None, data_false=None):
        self.value = value
        self.true = true
        self.false = false
        self.data_true = data_true
        self.data_false = data_false


class Tree:
    def __init__(self, data, algorithm, criteria):
        self.system_tf = [np.count_nonzero(data['Income'].to_numpy() == '<=50K'), np.count_nonzero(data['Income'].to_numpy() == '>50K')]
        self.total_length = self.system_tf[0] + self.system_tf[1]
        self.criteria = criteria
        self.algorithm = algorithm
        if self.criteria == 'e':
            self.system_entropy = self.calculate_system_entropy(self.system_tf)
        elif self.criteria == 'g':
            self.system_gini = self.calculate_system_gini(self.system_tf)
        self.attributes = data.columns
        self.data = data.to_numpy()
        self.data_true = self.data[self.data[:, - 1] == '<=50K']
        self.data_false = self.data[self.data[:, -1] == '>50K']
        self.node_list = deque()
        self.root = self.create_root()
        self.expand_tree(self.root)
        self.root.show_tree(level=0)

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
        return - true_part - false_part

    def calculate_system_gini(self, true_false):
        total_length = true_false[0] + true_false[1]
        if true_false[0] == 0:
            true_part = 0
        else:
            true_part = (true_false[0]/total_length) ** 2
        if true_false[1] == 1:
            false_part = 0
        else:
            false_part = (true_false[1]/total_length) ** 2
        return 1 - (true_part + false_part)

    def calculate_gini_attribute(self, attribute_array, node=None):
        if node is not None and node.inner_edge is not None:
            total_length = node.inner_edge.true + node.inner_edge.false
        else:
            total_length = self.total_length
        gini = 0.0
        for attribute_value in attribute_array[1]:
            total = attribute_value.true + attribute_value.false
            gini += (total / total_length) ** 2
        gini = 1 - gini
        return deque([attribute_array[0], gini])

    def calculate_entropy_attribute(self, attribute_array):
        entropy = 0.0
        for attribute_value in attribute_array[1]:
            total_length = attribute_value.true + attribute_value.false
            total_probability = total_length / self.total_length
            if attribute_value.true == 0:
                true_part = 0
            else:
                true_probability = attribute_value.true / total_length
                true_part = true_probability * math.log2(true_probability)
            if attribute_value.false == 0:
                false_part = 0
            else:
                false_probability = attribute_value.false / total_length
                false_part = false_probability * math.log2(false_probability)
            entropy += total_probability * (- true_part - false_part)
        return deque([attribute_array[0], entropy])

    def calculate_gain(self, att_entropy, node):
        return deque([att_entropy[0], node.entropy - att_entropy[1]])

    def calculate_split_info(self, attribute_array):
        split_info = 0.0
        for attribute_value in attribute_array[1]:
            total_length = attribute_value.true + attribute_value.false
            if total_length == 0 or self.total_length == 0:
                return [attribute_array[0], 0.0]
            total_probabilty = total_length / self.total_length
            split_info += (-total_probabilty * math.log2(total_probabilty))
        return deque([attribute_array[0], split_info])

    def calculate_gain_ratio(self, gain_array, split_array):
        gr = deque()
        for i, gain in enumerate(gain_array):
            if gain[1] == 0 or split_array[i][1] == 0:
                gr.append([gain[0], 0.0])
            else:
                gr.append([gain[0], gain[1] / split_array[i][1]])
        return gr

    def choose_winner(self, tf_array, node=None, edge=None):
        winner_entropy, winner_gini = None, None
        if self.algorithm == 'ID3':
            # Entropy
            if self.criteria == 'e':
                # Calculate the entropy of each attribute
                entropy_array, gain_array = deque(), deque()

                # Calculate the Entropy
                for attribute in tf_array:
                    entropy_array.append(self.calculate_entropy_attribute(attribute))

                # Calculate the Gain
                for entropy in entropy_array:
                    gain_array.append(self.calculate_gain(att_entropy=entropy, node=node))

                gains = deque()
                for gain in gain_array:
                    gains.append(gain[1])
                winner_index = gains.index(max(gains))
                winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
            else:
                # Gini
                gini_array = deque()
                # Calculate the Gini
                for attribute in tf_array:
                    gini_array.append(self.calculate_gini_attribute(attribute))

                ginis = deque()
                for gini in gini_array:
                    ginis.append(gini[1])
                winner_index = ginis.index(min(ginis))
                winner_attribute, winner_gini = gini_array[winner_index][0], gini_array[winner_index][1]
            winner_edges = deque()
            # Search the winner node's edges
            for true_false in tf_array:
                if true_false[0] == winner_attribute:
                    winner_edges = true_false[1]

            n_father_list = copy.deepcopy(node.father_list)
            n_father_list.append(winner_attribute)

            if self.criteria == 'e':
                winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, print=winner_attribute, edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
            else:
                winner_node = Node(gini=winner_gini, attribute=winner_attribute, print=winner_attribute, edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
            node.sons.append(winner_node)
            self.node_list.append(winner_node)
            return winner_node
        elif self.algorithm == 'C4.5':
            if self.criteria == 'e':
                # Calculate the entropy of each attribute
                entropy_array, gain_array, split_array = deque(), deque(), deque()
                # Calculate the Entropy
                for attribute in tf_array:
                    entropy_array.append(self.calculate_entropy_attribute(attribute))

                # Calculate the SplitInfo
                for attribute in tf_array:
                    split_array.append(self.calculate_split_info(attribute))

                # Calculate the Gain
                for entropy in entropy_array:
                    gain_array.append(self.calculate_gain(att_entropy=entropy, node=node))

                # Calculate GainRatio
                gr_array = self.calculate_gain_ratio(gain_array, split_array)

                grs = deque()
                for gr in gr_array:
                    grs.append(gr[1])
                winner_index = grs.index(min(grs))
                winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
            else:
                # Gini
                gini_array = deque()
                # Calculate the Gini and Gini Gain
                for attribute in tf_array:
                    gini_array.append(self.calculate_gini_attribute(attribute))
                gains = deque()
                for gain in gini_array:
                    gains.append(gain[1])
                winner_index = gains.index(min(gains))
                winner_attribute, winner_gini = gini_array[winner_index][0], gini_array[winner_index][1]

            winner_edges = deque()

            # Search the winner node's edges
            for true_false in tf_array:
                if true_false[0] == winner_attribute:
                    winner_edges = true_false[1]

            n_father_list = copy.deepcopy(node.father_list)
            n_father_list.append(winner_attribute)

            if self.criteria == 'e':
                winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, print=winner_attribute,
                                   edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
            else:
                winner_node = Node(gini=winner_gini, attribute=winner_attribute, print=winner_attribute,
                                   edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
            node.sons.append(winner_node)
            self.node_list.append(winner_node)
            return winner_node

    def calculate_true_false(self, index=None, node=None, edge=None):
        # We have to do two cases, the root case and the other ones
        true_false = deque()
        node_index = 0
        for i, v in enumerate(self.attributes):
            if v == node.attribute:
                node_index = i
                break
        # To save the attributes
        unique = np.unique(self.data[:, index])
        for attribute_value in unique:
            true_raw = np.count_nonzero(edge.data_true[:, index] == attribute_value)
            false_raw = np.count_nonzero(edge.data_false[:, index] == attribute_value)
            data_true = edge.data_true[edge.data_true[:, index] == attribute_value]
            data_false = edge.data_false[edge.data_false[:, index] == attribute_value]
            new_edge = Edge(value=attribute_value, true=true_raw, false=false_raw, data_true=data_true, data_false=data_false)
            true_false.append(new_edge)
        return true_false

    def create_root(self):
        tf_array = deque()
        for i in range(0, len(self.data[0]) - 1):
            true_false = deque()
            unique = np.unique(self.data[:, i])
            for attribute_value in unique:
                true_raw = np.count_nonzero(self.data_true[:, i] == attribute_value)
                false_raw = np.count_nonzero(self.data_false[:, i] == attribute_value)
                data_true = self.data_true[self.data_true[:, i] == attribute_value]
                data_false = self.data_false[self.data_false[:, i] == attribute_value]
                edge = Edge(value=attribute_value, true=true_raw, false=false_raw, data_true=data_true, data_false=data_false)
                true_false.append(edge)
            tf_array.append([self.attributes[i], true_false])

        winner_entropy, winner_gini = None, None
        if self.algorithm == 'ID3':
            if self.criteria == 'e':
                # Calculate the entropy of each attribute
                entropy_array = deque()
                # Calculate the Entropy
                for attribute in tf_array:
                    entropy_array.append(self.calculate_entropy_attribute(attribute))

                gains = deque()
                for entropy in entropy_array:
                    gains.append(entropy[1])
                winner_index = gains.index(min(gains))
                winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
            else:
                # Gini
                gini_array = deque()
                # Calculate the Gini and Gini Gain
                for attribute in tf_array:
                    gini_array.append(self.calculate_gini_attribute(attribute))
                gains = deque()
                for gain in gini_array:
                    gains.append(gain[1])
                winner_index = gains.index(min(gains))
                winner_attribute, winner_gini = gini_array[winner_index][0], gini_array[winner_index][1]

            winner_edges = deque()
            # Search the winner node's edges
            for true_false in tf_array:
                if true_false[0] == winner_attribute:
                    winner_edges = true_false[1]

            inner_edge = Edge(value='System', true=self.system_tf[0], false=self.system_tf[1])
            if self.criteria == 'e':
                root = Node(entropy=winner_entropy, attribute=winner_attribute, print=winner_attribute, edges=winner_edges, father=None, inner_edge=inner_edge, father_list=[winner_attribute])
            else:
                root = Node(gini=winner_gini, attribute=winner_attribute, print=winner_attribute, edges=winner_edges, father=None, inner_edge=inner_edge, father_list=[winner_attribute])
            root.root = True
            self.node_list.append(root)
            return root
        elif self.algorithm == 'C4.5':
            if self.criteria == 'e':
                # Calculate the entropy of each attribute
                entropy_array, gain_array, split_array = deque(), deque(), deque()
                # Calculate the Entropy
                for attribute in tf_array:
                    entropy_array.append(self.calculate_entropy_attribute(attribute))

                # Calculate the SplitInfo
                for attribute in tf_array:
                    split_array.append(self.calculate_split_info(attribute))

                # Calculate the Gain
                for entropy in entropy_array:
                    gain_array.append([entropy[0], self.system_entropy - entropy[1]])

                # Calculate GainRatio
                gr_array = self.calculate_gain_ratio(gain_array, split_array)

                grs = deque()
                for gr in gr_array:
                    grs.append(gr[1])
                winner_index = grs.index(min(grs))
                winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
            else:
                # Gini
                gini_array = deque()
                # Calculate the Gini and Gini Gain
                for attribute in tf_array:
                    gini_array.append(self.calculate_gini_attribute(attribute))
                gains = deque()
                for gain in gini_array:
                    gains.append(gain[1])
                winner_index = gains.index(min(gains))
                winner_attribute, winner_gini = gini_array[winner_index][0], gini_array[winner_index][1]
            winner_edges = deque()
            # Search the winner node's edges
            for true_false in tf_array:
                if true_false[0] == winner_attribute:
                    winner_edges = true_false[1]
            inner_edge = Edge(value='System', true=self.system_tf[0], false=self.system_tf[1])
            if self.criteria == 'e':
                root = Node(entropy=winner_entropy, attribute=winner_attribute, print=winner_attribute,
                            edges=winner_edges, father=None, inner_edge=inner_edge, father_list=[winner_attribute])
            else:
                root = Node(gini=winner_gini, attribute=winner_attribute, print=winner_attribute, edges=winner_edges,
                            father=None, inner_edge=inner_edge, father_list=[winner_attribute])
            root.root = True
            self.node_list.append(root)
            return root

    def check_edge(self, node, edge):
        for son in node.sons:
            if son.inner_edge == edge:
                return False
        return True

    def expand_tree(self, node):
        # Condicion de salida los datos del nodo són una decision
        if node.is_decision() or len(node.father_list) == (len(self.attributes) - 1):
            return

        for edge in node.edges:
            if self.check_edge(node, edge):
                tf_array = deque()
                for i in range(0, len(self.data[0]) - 1):
                    attribute = self.attributes[i]
                    if node.father is not None and attribute not in node.father_list:
                        tf_array.append([attribute, self.calculate_true_false(index=i, node=node, edge=edge)])
                    elif attribute != 'Income' and attribute not in node.father_list:
                        tf_array.append([attribute, self.calculate_true_false(index=i, node=node, edge=edge)])
                self.choose_winner(tf_array=tf_array, node=node, edge=edge)

        for son in node.sons:
            self.expand_tree(son)

    def predict(self, test):
        predictions = deque()
        node = self.root
        for i, row in test.iterrows():
            while not node.is_decision():
                for son in node.sons:
                    if son.inner_edge.value == row[node.attribute]:
                        node = son
            predictions.append(node.print)
        return np.array(predictions)

