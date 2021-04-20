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
        
class TruckTab:
    
    def __init__(self, name, quantity_max):
        self.name = name
        self.quantity_max = quantity_max
        self.remaining_quantity=quantity_max
        self.P = [0]
        self.cost = 0
        self.time = 0
            
    def calculate_cost(self, data,listOfClients):
        demande = 0
        weight_K = 10000
        weight_q = 10000
        weight_t = 3
        weight_d = 13
        weight_c = 10        
        for m in self.P:
            demande += listOfClients[m].quantity
        self.remaining_quantity-=demande
        if(self.remaining_quantity<0):
            self.cost+=(self.remaining_quantity*(-1))*weight_q
        for i in range (len(self.P)-1):
            self.cost += (distanceTab(self.P[i],self.P[i+1],listOfClients)*weight_c)
            self.time += data[self.P[i]][self.P[i+1]]
            if self.time <= listOfClients[self.P[i+1]].start:
                self.cost += ((listOfClients[self.P[i+1]].start - self.time)*weight_t)
                self.time = listOfClients[self.P[i+1]].start
            if self.time >= listOfClients[self.P[i+1]].stop:
                self.cost += ((self.time - listOfClients[self.P[i+1]].stop)*weight_d)
        self.cost += weight_K
        return (self.cost)
def distanceTab(C1, C2, listOfClients):
        return(np.sqrt((listOfClients[C1].y-listOfClients[C2].y)**2 + (listOfClients[C1].x - listOfClients[C2].x)**2))
