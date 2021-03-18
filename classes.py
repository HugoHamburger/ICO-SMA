#On a ici défini une classe Client commune à nos trois algorithmes 
#afin d'encoder nos clients à livrer


class Client:
    def __init__(self, name, x, y, quantity, start, stop):
        self.name = name
        self.x = x
        self.y = y
        self.quantity = quantity
        self.start = start
        self.stop = stop
        self.delivered = False
