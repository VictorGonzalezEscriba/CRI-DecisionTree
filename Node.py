

class Node:
    def __init__(self, entropy=None, father=None, edges=None, attribute=None, inner_edge=None, root=None, father_attribute=None, father_list=None):
        self.root = root
        self.father = father
        self.father_attribute = father_attribute
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
        self.create_leaf()
        self.father_list = father_list

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
        son = Node(father=self, inner_edge=inner_edge, attribute=decision)
        son.set_leaf()
        self.add_son(son)

    def add_son(self, son):
        self.sons.append(son)

    # To see if the node has a leaf son
    def create_leaf(self):
        if self.edges is not None:
            for edge in self.edges:
                # To avoid {0,0}
                if edge[1] == 0 and edge[2] == 0:
                    pass
                else:
                    if edge[1] == 0:
                        self.create_leaf_son(edge, '>50K')
                    elif edge[2] == 0:
                        self.create_leaf_son(edge, '<=50K')

    def show_tree(self, level=0):
        if len(self.inner_edge) != 2:
            print('\t' * level + '|_', '(' + self.attribute + ')', ' - ', self.inner_edge[0], '{', self.inner_edge[1], ',', self.inner_edge[2], '}')
        else:
            print('\t' * level + '(' + self.attribute + ')', ' - ', '{', self.inner_edge[0], ',', self.inner_edge[1], '}')
        for son in self.sons:
            son.show_tree(level+1)
