# -*- coding: utf-8 -*-
"""
SMA Collaboration

"""


from mesa import Agent
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector



    
class gen_agent(Agent):
    
    def __init__(self):
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution
       
    
    

        
        
        
class tab_agent(Agent):
    
    def __init__(self):
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution
        
        
        
        
        
class rs_agent(Agent):

    def __init__(self):
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution


    ################################## Utility functions ##################################

    def find_client_by_name(name,C):
        i = 0
        n = len(C)
        find = False
        client = Client(-1,0,0,0,0,0)
        while i < n and find == False : 
            if C[i].name == name :
                client = C[i]
                find = True
            i +=1
        return client

    def distance(C1, C2):
        return(np.sqrt((C1.y-C2.y)**2 + (C1.x - C2.x)**2))

    def cost_matrix(C):
        n = len(C)
        i=0
        j=0
        M = np.zeros((n+1,n+1))
        for i in range(1,n+1):
            M[i][0] = np.sqrt(C[i-1].x**2+C[i-1].y**2)
            M[0][i] = np.sqrt(C[i-1].x**2+C[i-1].y**2)
            for j in range(1,n+1):
                M[i][j] = distance(C[i-1],C[j-1])
        return M

    def quantity(C):
        q = 0
        for i in range(len(C)):
            q += C[i].quantity
        return q

    def copy(F):
        L = []
        for x in F:
            V = []
            for i in range(len(x)):
                V.append(x[i])
            L.append(V)
        return L

    def closing_tour(itineraire): 
        L = []
        for x in itineraire:
            if len(x)>1:
                x.append(0)
                del(x[0])
                x.insert(0,0)
            else:
                L.append(x)
        for x in L:
            itineraire.remove(x)  

    def convert_solution(solution):
        if solution[0][0]==0:
            for i in range(len(solution)):
                solution[i][0]=Truck(i,truck_capacity,0,0)
                del(solution[i][-1])
            for i in range(len(solution),trucks_disponibility+1):
                solution.append(Truck(i,truck_capacity,0,0))
        return solution

    #################################### RS Algorithm #####################################
    def random_solution(C):
        C2 = [client.name for client in C]
        q = quantity(C)
        t = (q // truck_capacity) + 2
        T = [Truck(i,truck_capacity,0,0) for i in range(trucks_disponibility)]
        R = []
        C_bis = C2.copy()
        for i in range(t-1):
            s = rd.sample(C_bis,rd.randint(1,len(C_bis)-(t-i+1)))
            R.append([T[i]]+s)
            for x in s :
                C_bis.remove(x)
        rd.shuffle(C_bis)
        L  = [T[t-1]]+C_bis
        R.append(L)
        for i in range(t,len(T)):
            R.append([T[i]])
        return R

    def neighbouring_solution(R):
         n = len(R)
         S = copy(R)
         i = rd.randint(0,n-1)
         j = rd.randint(0,n-1)
         l = rd.randint(1,len(S[i]))
         k = rd.randint(1,len(S[j]))
         if l == len(S[i]) and k == len(S[j]):
             return S
         elif k == len(S[j]) :
             client = S[i][l]
             S[j].append(client)
             S[i].remove(client)
         elif l == len(S[i]) : 
            client = S[j][k]
            S[i].append(client)
            S[j].remove(client)
         else:         
             client1 = S[i][l]
             client2 = S[j][k]
             S[i][l] = client2
             S[j][k] = client1
         return S

    def cost_function(R,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c):
        T = 0
        n = len(R)
        K = 0
        for x in R :
            if len(x)> 1 :
                K+=1
        t = 0
        d = 0
        c = 0
        q = 0
        dispo = [R[i][0].start for i in range(n) ]
        for i in range(n):
            if len(R[i])> 1 :
                truck = R[i][0]
                truck.remaining_quantity = truck.quantity_max
                client = find_client_by_name(R[i][1],C)
                dispo[i] = dispo[i] + M_time[client.name][0]
                c+= M[client.name][0]
                d += max(0,dispo[i] - client.stop)
                t -= min(0, dispo[i] - client.start)
                truck.delivery(client)
                for j in range(2,len(R[i])):
                    client_prec = find_client_by_name(R[i][j-1],C)
                    client = find_client_by_name(R[i][j],C)
                    c+= M[client_prec.name][client.name]
                    dispo[i] = dispo[i] + M_time[client_prec.name][client.name]
                    d += max(0,dispo[i] - client.stop)
                    t -= min(0, dispo[i] - client.start)
                    truck.delivery(client)
                c += M[0][client.name]
                q -= min(0,truck.remaining_quantity)
        T = weight_K * K + weight_q * q + weight_t * t + weight_d * d + weight_c * c
        return T

    def algo_RS(C,M_time,start_solution = random_solution(C), n=100_000, weight_K=10_000, weight_q=10_000, weight_t=3, weight_d=13, weight_c=10):
        M = cost_matrix(C)
        R = convert_solution(start_solution)
        T = cost_function(R,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c)
        i = 0
        while  i < n :
            S = neighbouring_solution(R)
            T_bis = cost_function(S,M,M_time,C, weight_K, weight_q, weight_t, weight_d, weight_c)
            p = (T_bis - T)*10000
            random = rd.random()
            if p<0 or random < np.exp(-p/T):
                R = copy(S)
                T = T_bis
            i+= 1
        return (R,T_bis)

    ###################################### Indicators #######################################
    def delay_indicator(R,C,M_time):
        dispo = [0]* len(R)
        delay_by_truck = [0]* len(R)
        clients_name = []
        cumulative_delay = 0
        for x in R:
            if len(x)>1:
                c= find_client_by_name(x[1],C)
                dispo[x[0].name] += M_time[c.name][0]
                if dispo[x[0].name]-c.stop > 0 :
                    delay_by_truck[x[0].name] += dispo[x[0].name]-c.stop
                    clients_name.append(c.name)
                    cumulative_delay += dispo[x[0].name]-c.stop
                for i in range(2,len(x)) :
                    c= find_client_by_name(x[i],C)
                    dispo[x[0].name] += M_time[c.name][x[i-1]]
                    if dispo[x[0].name]-c.stop > 0 :
                        clients_name.append(c.name)
                        delay_by_truck[x[0].name] += dispo[x[0].name]-c.stop
                        cumulative_delay +=dispo[x[0].name]-c.stop
        return delay_by_truck, clients_name, cumulative_delay


    def advance_indicator(R,C,M_time):
        dispo = [0]* len(R)
        advance_by_truck = [0] * len(R)
        cumulative_advance = 0
        for x in R:
            if len(x)>1:
                c= find_client_by_name(x[1],C)
                dispo[x[0].name] += M_time[c.name][0]
                if c.start - dispo[x[0].name] > 0 :
                    advance_by_truck[x[0].name] += c.start - dispo[x[0].name]
                    cumulative_advance += c.start - dispo[x[0].name]
                for i in range(2,len(x)) :
                    c= find_client_by_name(x[i],C)
                    dispo[x[0].name] += M_time[c.name][x[i-1]]
                    if dispo[x[0].name]-c.stop > 0 :
                        advance_by_truck[x[0].name] +=c.start - dispo[x[0].name]
                        cumulative_advance +=c.start - dispo[x[0].name]
        return advance_by_truck, cumulative_advance

    ###################################### Step #######################################

    def step(self):
        self.solution = closing_tour(algo_RS(self.model.list_clients, self.model.time_matrix)[0])
        #prendre un solution aléatoirement dans le pool
        #for a in self.model.schedule.agents:
        #    if isinstance(a,pool_agent):
        #        a.pool
        
        
class pool_agent(Agent):
    def __init__(self, id, model, nb_solutions,pool_radius):
        super().__init__(id,model)
        self.nb_solutions = nb_solutions
        self.pool = []
        self.pr = pool_radius
        
    def distance(self,solution1,solution2):
        s1 = []
        s2=[]
        r = 0
        for x in solution1:
            for i in range(1,len(x)):
                s1.append((x[i-1],x[i]))
        for x in solution2:
            for i in range(1,len(x)):
                s2.append((x[i-1],x[i]))
        for x in s1 :
            if not x in s2:
                r += 1
        return 1-r/self.pr if r<self.pr else 0
    
    
    def eval_function(self):
        pool = self.pool
        g = 0
        for i in range(len(pool)-1) :
            for y in pool[i+1:]:
                g+= self.distance(y,pool[i])
                
                
    def solution(self):
        for a in self.model.schedule.agents:
            if isinstance(a,tab_agent) or isinstance(a,rs_agent) or isinstance(a,gen_agent):
                solution = a.solution
                res = -1
                if len(self.pool)< self.nb_solutions :
                    phi = 0
                    for x in self.pool:
                        phi += self.distance(x,solution)
                    if phi == 0:
                        self.pool.append(solution)
                        break
                else :
                    pool = self.pool
                    g = self.eval_function(pool)
                    for i in range(pool):
                        pool_test = pool.copy()
                        pool_test[i] = solution
                        g_bis = self.eval_function(pool_test)
                        if g_bis < g :
                            g = g_bis
                            res = i
                            
                    if res != -1 :
                        self.pool[i]=solution
                        
    def step(self):
        self.solution()
        

        
        
class graphic_agent(Agent):
    
    def __init__(self):        
 
        
        
        
        

#la classe SMA
class SMA_collab(Model):
    """A model for infection spread."""

    def __init__(self, n_truck, truck_capacity, list_clients, time_matrix, n_pool):

        self.n_truck = n_truck
        self.truck_capacity = truck_capacity
        self.list_clients = list_clients
        self.time_matrix = time_matrix
        self.n_pool = n_pool



        
        
        #l'ordonnanceur du modele (instance de RandomActivation)
        #tester SimultaneousActivation (qui permet d'activer tous les agents en mêê temps)
        self.schedule = BaseScheduler(self)

        

        a = gen_agent(...)
        self.schedule.add(a)
        
        a = tab_agent(...)
        self.schedule.add(a)
        
        a = rs_agent(...)
        self.schedule.add(a)
        
        
        # Gestion du pool
        a = pool_agent()
        self.schedule.add(a)
        
        
        # Gestion des courbes
        a = graphic_agent()
        self.schedule.add(a)
        
        
    def step(self):
        #passage de l'instant t à l'instant (t+1)
        self.schedule.step()
        


        
