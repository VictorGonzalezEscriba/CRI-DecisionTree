class Node:
    def _init_(self, _padre, _datos, _true_false):
        self.padre = _padre
        self.datos = _datos
        self.true_false = _true_false
        self.hijos = []
        self.decision = None  # serà <=50k o al reves en caso de que sea decision
        self.system_entropy = 0

    def isDecision(self):
        # Aqui programar que mire los datos i si són one-sided retornar true i assignar decision
        return False

    def calc_system_entropy(self):
        self.system_entropy = 0  #
        # la retornamos tmb
        return self.system_entropy
        pass



def id3(node):
    # Condicion de salida los datos del nodo són una decision
    if node.isDecision():
        return True
    else:
        # Entonces expandimos a hijos

        # Calculamos entropia del sistema
        s_entropy = node.calc_system_entropy()

        # escogemos ganador
        ganador = "Lo que sea"

        # Recortamos data set
        d = node.datos[:]  # quitamos atributo ganador
        tf = [0, 0]  # calculamos tf
        for arista in ganador:
            node.hijos.append(Node(node, d, tf))

        # expandimos
        for h in node.hijos:
            id3(h)