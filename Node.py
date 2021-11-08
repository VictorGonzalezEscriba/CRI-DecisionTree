import numpy as np


class Node:
    def __init__(self, entropy=None, num_samples=None, true_false=None, father=None, id=None, edge=None, data=None):
        self.id = id
        self.father = father
        self.sons = []

        # The question (?)
        self.edge = edge

        # Value of the node's entropy
        self.entropy = entropy
        self.data = data
        self.visited = False
        # To calculate the entropy, in 3{2+,1-}
        # 3 is num_samples
        # 2,1 are the two positions of true_false
        self.num_samples = num_samples
        self.true_false = true_false

        # Attributes for the final decision
        self.leaf = False
        self.decision = ''

    def print_node(self):
        print("Id: ", self.id)
        print("Father: ", self.father)
        print("Edge: ", self.edge)
        print("Sons: ", self.sons)
        print("N Samples:", self.num_samples)
        print("True-False:", self.true_false)
        print("Entropy: ", self.entropy)

    # Getters
    def get_id(self):
        return self.id

    def get_num_samples(self):
        return self.num_samples

    def get_true_false(self):
        return self.true_false

    def get_entropy(self):
        return self.entropy

    # Setters
    def set_entropy(self, entropy):
        self.entropy = entropy

    def set_visited_true(self):
        self.visited = True

    # Method
    def add_son(self, son):
        self.sons.append(son)
