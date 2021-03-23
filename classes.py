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
     
    def calculate_cost(self, data,listOfClients):
        demande = 0
        for m in self.P:
            demande += listOfClients[m].quantity
        self.remaining_quantity-=demande
        if(self.remaining_quantity<0):
            self.cost+=self.remaining_quantity*(-1)
        for i in range (len(self.P)-1):
            self.cost += distance (self.P[i],self.P[i+1],listOfClients)
            self.time += data['time_matrix'][self.P[i]][self.P[i+1]]
            if self.time <= listOfClients[self.P[i+1]].start:
                self.time = listOfClients[self.P[i+1]].start
            if self.time >= listOfClients[self.P[i+1]].stop:
                self.cost += self.time - listOfClients[self.P[i+1]].stop
        return self.cost
