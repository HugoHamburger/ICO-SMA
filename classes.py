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

class Truck:    
    def __init__(self, name, quantity_max, start, stop):
        self.name = name
        self.x = 0
        self.y = 0
        self.quantity_max = quantity_max
        self.start = start
        self.stop = stop
        self.remaining_quantity=quantity_max
        self.vitesse = 25

    def delivery(self, client):
        self.x=client.x
        self.y=client.y
        self.remaining_quantity -= client.quantity
