

class Node:
    def __init__(self, entropy=None, father=None, edges=None, attribute=None, inner_edge=None, root=None):
        self.root = root
        self.father = father
        self.sons = []
        # The question
        self.inner_edge = inner_edge
        self.edges = edges
        # Value of the node's entropy
        self.entropy = entropy
        # The attribute we set to the node
        self.attribute = attribute
        self.visited = False
        # Attributes for the final decision
        self.leaf = False
        self.decision = ''
        self.has_a_leaf()


    def print_node(self):
        print("Father: ", self.father)
        print("Inner Edge: ", self.inner_edge)
        print("Edges: ", self.edges)
        print("Sons: ", self.sons)
        print("Entropy: ", self.entropy)

    # Getters
    def get_entropy(self):
        return self.entropy

    def get_attribute(self):
        return self.attribute

    # Setters
    def set_entropy(self, entropy):
        self.entropy = entropy

    def set_attribute(self, attribute):
        self.attribute = attribute

    def set_visited_true(self):
        self.visited = True

    def set_leaf(self):
        self.leaf = True

    def set_decision(self, decision):
        self.decision = decision

    # Methods
    def create_leaf_son(self, inner_edge, decision):
        son = Node(father=self, inner_edge=inner_edge)
        son.set_decision(decision)
        self.add_son(son)

    def add_son(self, son):
        self.sons.append(son)

    # To see if the node has a leaf son
    def has_a_leaf(self):
        for edge in self.edges:
            if edge[1] == 0:
                self.create_leaf_son(edge, '<=50K')
                return True
            elif edge[2] == 1:
                self.create_leaf_son(edge, '>50K')
                return True
        return False
