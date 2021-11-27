import numpy as np
import math
import copy


class Node:
    def __init__(self, father=None, attribute=None, entropy=None, edges=None, inner_edge=None, father_list=None):
        self.father = father
        self.sons = []
        self.attribute = attribute
        self.entropy = entropy
        self.edges = edges
        self.inner_edge = inner_edge
        self.decision = ''
        self.root = False
        self.leaf = False
        self.father_list = father_list
        self.create_leaf()

    def show_tree(self, level=0):
        if not self.root:
            print('\t' * level + '|_', '(' + self.attribute + ')', ' - ', self.inner_edge.value, '{', self.inner_edge.true, ',', self.inner_edge.false, '}')
        else:
            print('\t' * level + '(' + self.attribute + ')', ' - ', '{', self.inner_edge.true, ',', self.inner_edge.false, '}')
        for son in self.sons:
            son.show_tree(level+1)

    def create_leaf_son(self, inner_edge, decision):
        son = Node(father=self, inner_edge=inner_edge, attribute=decision)
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
                    else:
                        self.create_leaf_son(edge, '>50K')
                else:
                    if edge.true == 0:
                        self.create_leaf_son(edge, '>50K')
                    elif edge.false == 0:
                        self.create_leaf_son(edge, '<=50K')

    def is_decision(self):
        if self.attribute == '>50K' or self.attribute == '<=50K':
            return True
        else:
            return False


class Edge:
    def __init__(self, value=None, true=None, false=None, data_true = None, data_false = None):
        self.value = value
        self.true = true
        self.false = false
        self.data_true = data_true
        self.data_false = data_false


class ID3:
    def __init__(self, data):
        self.system_tf = [np.count_nonzero(data['Income'].to_numpy() == '<=50K'), np.count_nonzero(data['Income'].to_numpy() == '>50K')]
        self.system_data_true = data['Income'] == '<=50K'
        self.system_data_false = data['Income'] == '>50K'
        self.total_length = data.shape[0]
        self.system_entropy = self.calculate_system_entropy(self.system_tf)
        self.data = data
        self.root = self.create_root(data)
        self.id3(self.root)
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

    def calculate_gain(self, att_entropy, node):
        return [att_entropy[0], node.entropy - att_entropy[1]]

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
        return [attribute_array[0], entropy]

    def choose_winner(self, tf_array, node=None, edge=None):
        # Calculate the entropy of each attribute
        entropy_array, gain_array = [], []
        # Calculate the Entropy
        for attribute in tf_array:
            entropy_array.append(self.calculate_entropy_attribute(attribute))
        # Calculate the Gain
        if len(entropy_array) != 0:
            for entropy in entropy_array:
                gain_array.append(self.calculate_gain(att_entropy=entropy, node=node))
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
            n_father_list = copy.deepcopy(node.father_list)
            n_father_list.append(winner_attribute)
            winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
            node.sons.append(winner_node)
            return winner_node

    def calculate_true_false(self, data=None, attribute=None, node=None, edge=None):
        # We have to do two cases, the root case and the other ones
        true_false = []
        # To save the attributes
        unique = np.unique(data[attribute].to_numpy())
        for attribute_value in unique:
            if node.root:
                true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K') & (data[node.attribute] == edge.value)]
                false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K') & (data[node.attribute] == edge.value)]
            else:
                true_raw = edge.data_true.loc[(edge.data_true[attribute] == attribute_value) & (edge.data_true['Income'] == '<=50K') & (edge.data_true[node.attribute] == edge.value)]
                false_raw = edge.data_false.loc[(edge.data_false[attribute] == attribute_value) & (edge.data_false['Income'] == '>50K') & (edge.data_false[node.attribute] == edge.value)]
            new_edge = Edge(value=attribute_value, true=true_raw.shape[0], false=false_raw.shape[0], data_true=true_raw, data_false=false_raw)
            true_false.append(new_edge)
        return true_false

    def create_root(self, data):
        tf_array = []
        for attribute in data.columns:
            true_false = []
            if attribute != 'Income':
                unique = np.unique(data[attribute].to_numpy())
                for attribute_value in unique:
                    true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K')]
                    false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K')]
                    edge = Edge(value=attribute_value, true=true_raw.shape[0], false=false_raw.shape[0])
                    true_false.append(edge)
                tf_array.append([attribute, true_false])
        # Calculate the entropy of each attribute
        entropy_array, gain_array = [], []
        # Calculate the Entropy
        for attribute in tf_array:
            entropy_array.append(self.calculate_entropy_attribute(attribute))
        # Calculate the Gain
        if len(entropy_array) != 0:
            for entropy in entropy_array:
                gain_array.append([entropy[0], self.system_entropy - entropy[1]])
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
            inner_edge = Edge(value='System', true=self.system_tf[0], false=self.system_tf[1], data_true=self.system_data_true, data_false=self.system_data_false)
            root = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, father=None, inner_edge=inner_edge, father_list=[winner_attribute])
            root.root = True
            return root

    def check_edge(self, node, edge):
        for son in node.sons:
            if son.inner_edge == edge:
                return False
        return True

    def id3(self, node):
        # Condicion de salida los datos del nodo són una decision
        if node.is_decision():
            return

        for edge in node.edges:
            if self.check_edge(node, edge):
                tf_array = []
                for attribute in self.data.columns:
                    if node.father is not None and attribute != 'Income' and attribute not in node.father_list:
                        tf_array.append([attribute, self.calculate_true_false(data=self.data, attribute=attribute, node=node, edge=edge)])
                    elif attribute != 'Income' and attribute not in node.father_list:
                        tf_array.append([attribute, self.calculate_true_false(data=self.data, attribute=attribute, node=node, edge=edge)])
                self.choose_winner(tf_array=tf_array, node=node, edge=edge)

        for son in node.sons:
            self.id3(son)


class C45:
    def __init__(self, data):
        self.system_tf = [np.count_nonzero(data['Income'].to_numpy() == '<=50K'), np.count_nonzero(data['Income'].to_numpy() == '>50K')]
        self.system_data_true = data['Income'] == '<=50K'
        self.system_data_false = data['Income'] == '>50K'
        self.total_length = data.shape[0]
        self.system_entropy = self.calculate_system_entropy(self.system_tf)
        self.data = data
        self.root = self.create_root(data)
        self.c45(self.root)
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

    def calculate_split_info(self, attribute_array):
        split_info = 0.0
        for attribute_value in attribute_array[1]:
            total_length = attribute_value.true + attribute_value.false
            if total_length == 0 or self.total_length == 0:
                return [attribute_array[0], 0.0]
            total_probabilty = total_length / self.total_length
            split_info += (-total_probabilty * math.log2(total_probabilty))
        return [attribute_array[0], split_info]
    
    def calculate_gain_ratio(self, gain_array, split_array):
        gr = []
        for i, gain in enumerate(gain_array):
            if gain[1] == 0 or split_array[i][1] == 0:
                gr.append([gain[0], 0.0])
            else:
                gr.append([gain[0], gain[1] / split_array[i][1]])
        return gr

    def calculate_gain(self, att_entropy, node):
        return [att_entropy[0], node.entropy - att_entropy[1]]

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
        return [attribute_array[0], entropy]

    def choose_winner(self, tf_array, node=None, edge=None):
        # Calculate the entropy of each attribute
        entropy_array, gain_array, split_array = [], [], []
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

        grs = []
        for gr in gr_array:
            grs.append(gr[1])
        winner_index = grs.index(max(grs))
        winner_attribute, winner_entropy = entropy_array[winner_index][0], entropy_array[winner_index][1]
        winner_edges = []

        # Search the winner node's edges
        for true_false in tf_array:
            if true_false[0] == winner_attribute:
                winner_edges = true_false[1]

        n_father_list = copy.deepcopy(node.father_list)
        n_father_list.append(winner_attribute)
        if node.father is not None:
            edge.data_true = edge.data_true.drop([node.father.attribute], axis=1)
            edge.data_false = edge.data_false.drop([node.father.attribute], axis=1)
        winner_node = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, father=node, inner_edge=edge, father_list=n_father_list)
        node.sons.append(winner_node)
        return winner_node

    def calculate_true_false(self, data=None, attribute=None, node=None, edge=None):
        # We have to do two cases, the root case and the other ones
        true_false = []
        # To save the attributes
        unique = np.unique(data[attribute].to_numpy())
        for attribute_value in unique:
            if node.root:
                true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K') & (                data[node.attribute] == edge.value)]
                false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K') & (                data[node.attribute] == edge.value)]
            else:
                true_raw = edge.data_true.loc[(edge.data_true[attribute] == attribute_value) & (edge.data_true['Income'] == '<=50K') & (edge.data_true[node.attribute] == edge.value)]
                false_raw = edge.data_false.loc[(edge.data_false[attribute] == attribute_value) & (edge.data_false['Income'] == '>50K') & (edge.data_false[node.attribute] == edge.value)]
            new_edge = Edge(value=attribute_value, true=true_raw.shape[0], false=false_raw.shape[0],data_true=true_raw, data_false=false_raw)
            true_false.append(new_edge)
        return true_false

    def create_root(self, data):
        tf_array = []
        for attribute in data.columns:
            true_false = []
            if attribute != 'Income':
                unique = np.unique(data[attribute].to_numpy())
                for attribute_value in unique:
                    true_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '<=50K')]
                    false_raw = data.loc[(data[attribute] == attribute_value) & (data['Income'] == '>50K')]
                    edge = Edge(value=attribute_value, true=true_raw.shape[0], false=false_raw.shape[0])
                    true_false.append(edge)
                tf_array.append([attribute, true_false])
        # Calculate the entropy of each attribute
        entropy_array, gain_array = [], []
        # Calculate the Entropy
        for attribute in tf_array:
            entropy_array.append(self.calculate_entropy_attribute(attribute))
        # Calculate the Gain
        if len(entropy_array) != 0:
            for entropy in entropy_array:
                gain_array.append([entropy[0], self.system_entropy - entropy[1]])
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
            inner_edge = Edge(value='System', true=self.system_tf[0], false=self.system_tf[1],data_true=self.system_data_true, data_false=self.system_data_false)
            root = Node(entropy=winner_entropy, attribute=winner_attribute, edges=winner_edges, father=None,inner_edge=inner_edge, father_list=[winner_attribute])
            root.root = True
            return root

    def check_edge(self, node, edge):
        for son in node.sons:
            if son.inner_edge == edge:
                return False
        return True

    def c45(self, node):
        # Condicion de salida los datos del nodo són una decision
        if node.is_decision() or len(node.father_list) == (len(self.data.columns) - 1):
            return

        for edge in node.edges:
            if self.check_edge(node, edge):
                tf_array = []
                for attribute in self.data.columns:
                    if node.father is not None and attribute != 'Income' and attribute not in node.father_list:
                        tf_array.append([attribute,self.calculate_true_false(data=self.data, attribute=attribute, node=node,edge=edge)])
                    elif attribute != 'Income' and attribute not in node.father_list:
                        tf_array.append([attribute,self.calculate_true_false(data=self.data, attribute=attribute, node=node,edge=edge)])
                self.choose_winner(tf_array=tf_array, node=node, edge=edge)

        for son in node.sons:
            self.c45(son)

