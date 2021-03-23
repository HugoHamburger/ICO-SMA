# -*- coding: utf-8 -*-
"""
SMA Collaboration

"""
from classes.py import Client, Truck

from mesa import Agent
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from evolution_functions import next_gen, init_pop, merge_sort
from constants import nb_generations, nb_pop, n_trucks, truck_capacity, mutation_rate, list_clients
from evaluation_functions import truck_track_constructor, track_to_member
import matplotlib.pyplot as plt
import random as rd
import numpy as np

    
class gen_agent(Agent):
    
    def __init__(self, nb_pop, nb_generations, n_truck, truck_capacity, list_clients):
        self.nb_pop = nb_pop
        self.nb_generations =  nb_generations
        self.n_trucks = n_trucks
        self.truck_capacity = truck_capacity
        self.list_clients = list_clients
        self.time_matrix = time_matrix     
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution        
        self.population = init_pop(nb_pop)
        self.solution = self.population[0]
    
    def step(self):
        for i in range(2,nb_generations+1):
            self.population = next_gen(self.population)
        self.solution = self.population[0]
                
class tab_agent(Agent):
    class Depot:
        def __init__(self, num_truck):
            self.x = 0
            self.y = 0
            self.num_truck = num_truck
            self.T = []
            for i in range (num_truck):
                truck = Truck (i, self.model.truck_capacity)
                self.T.append(truck)

    def __init__(self):
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution
        
    def step(self):
        sol =(algo_tabou (200, 20,self.model.n_truck,self.model.list_clients))
        self.solution = sol[0]
        
    def distance(C1, C2, listOfClients):
        return(np.sqrt((listOfClients[C1].y-listOfClients[C2].y)**2 + (listOfClients[C1].x - listOfClients[C2].x)**2))

    def solution_initiale (num_truck,listOfClients,data):
        for a in self.model.schedule.agents:
            if isinstance(a,pool_agent):
                if len(a.pool) ==a.nb_solutions:
                    i = rd.randint(0, a.nb_solutions-1)
                    return a.pool[i]
        S=[]
        depot = Depot(num_truck)
        clients = [i for i in range (1,len(data['time_matrix']))]
        while len (clients) != 0:
            for k in range (len (depot.T)):
                if len (clients) != 0:
                    j = rd.randint (0,len(clients)-1)
                    depot.T[k].P.append(clients[j])
                    clients.remove(clients[j])

        for k in range (len(depot.T)):
            depot.T[k].P.append(0)
        for i in range (num_truck):
            S.append(depot.T[i].P)
        return (total_cost(depot,data,listOfClients),S)
    
                    
    def total_cost (depot, data,listOfClients):
        total_cost = 0
        for i in range (depot.num_truck):
            (depot.T[i]).calculate_cost(data,listOfClients)
            total_cost += depot.T[i].cost
        return total_cost

    def simple_permut (P, old, new):
        temp = P[old]
        P.remove(P[old])
        P.insert(new,temp)
        return P

    def voisinage_simple (parcours, num_truck, data,listOfClients):
        copie_parcours=parcours+[]
        vs=[parcours]
        truck= Truck(num_truck+1,15)
        truck.P = parcours
        cost=[truck.calculate_cost(data,listOfClients)]
        for i in range (1,len(copie_parcours)-1):
            for j in range (1,len(copie_parcours)-1):
                if i < j:
                    truck = Truck(num_truck+2,15)
                    copie_parcours=simple_permut(copie_parcours,i,j)
                    vs.append(copie_parcours)
                    truck.P = copie_parcours
                    cost.append(truck.calculate_cost(data,listOfClients))
                    copie_parcours=parcours+[]
        mini=cost[0]
        index_opti=0
        for k in range (1,len(cost)):
            if cost[k]<mini:
                mini=cost[k]
                index_opti=k
        return vs[index_opti]

    
    def voisinage_complexe (ens_parcours, num_truck, data,tabou,listOfClients):
        copie_ens_parcours=ens_parcours+[]
        vc=[]
        list_tot_cost=[]
        for i in range (len(ens_parcours)):
            for j in range (len(ens_parcours)):
                if i != j:
                    for k in range (1,len(copie_ens_parcours[i])-1):
                        for l in range (1,len(copie_ens_parcours[j])-1):
                            (a,b) = transfert(copie_ens_parcours[i],copie_ens_parcours[j],k,l)
                            ens_parcours_to_add=[]
                            depot0 = Depot (num_truck)
                            for w in range (len(copie_ens_parcours)):
                                if w == i :
                                    ens_parcours_to_add.append(a)
                                    depot0.T[w].P=a
                                elif w == j :
                                    ens_parcours_to_add.append(b)
                                    depot0.T[w].P=b
                                else:
                                    ens_parcours_to_add.append(copie_ens_parcours[w])
                                    depot0.T[w].P=copie_ens_parcours[w]
                            vc.append(ens_parcours_to_add)
                            list_tot_cost.append(total_cost(depot0,data,listOfClients))
        return (list_tot_cost, vc)                                      
                    
            
    def best_voisinage (num_truck, data,sol_actuelle,tabou,best_saved_cost,listOfClients):
        all_ens_vs=[]
        all_cost=[]
        for i in range (len(sol_actuelle)):
            ens_vs=[]
            cost=0
            for j in range (len(sol_actuelle)):
                if i==j:
                    truck = Truck(num_truck+1, 15)
                    truck.P = voisinage_simple(sol_actuelle[i], num_truck, data,listOfClients)
                    ens_vs.append(truck.P)
                    cost+= truck.calculate_cost(data,listOfClients)
                else:
                    truck = Truck(num_truck+1, 15)
                    truck.P = sol_actuelle[j]
                    cost+= truck.calculate_cost(data,listOfClients)
                    ens_vs.append(sol_actuelle[j])
            all_ens_vs.append(ens_vs)
            all_cost.append(cost)

        (cost_vc,vc)=voisinage_complexe(sol_actuelle,num_truck,data,tabou,listOfClients)
        all_ens_vs = all_ens_vs+vc
        all_cost = all_cost+cost_vc
        (all_ens_vs,all_cost) = sol_filter(data,sol_actuelle,tabou,all_ens_vs,all_cost,best_saved_cost)
        if(len(all_ens_vs)>0):
            min_cost=all_cost[0]
            index_best_cost=0
            for i in range (len(all_cost)):
                if (all_cost[i] < min_cost):
                    min_cost = all_cost[i]
                    index_best_cost = i
            return (all_cost[index_best_cost],all_ens_vs[index_best_cost])
        else:
            return (0,[])
    
    def sol_filter(data,sol_actuelle,tabou,all_ens_vs,all_cost,best_saved_cost):
        new_sol =[]
        new_sol_cost=[]
        for i in range (len(all_ens_vs)):
            if((sol_actuelle,all_ens_vs[i])not in tabou or all_cost[i]<best_saved_cost):
                new_sol.append(all_ens_vs[i])
                new_sol_cost.append(all_cost[i])
        return (new_sol,new_sol_cost)

    def transfert (P1, P2, old_pos, new_pos):
        copy_P1=P1+[]
        copy_P2=P2+[]
        temp = copy_P1[old_pos]
        copy_P1.remove(copy_P1[old_pos])
        copy_P2.insert(new_pos,temp)
        return (copy_P1,copy_P2)  

     def algo_tabou (nb_iter, max_tabou_size,number_trucks,listOfClients):
        curr_cost =0
        curr_sol=[]
        tabou =[]
        data = create_data_model()
        (curr_cost,curr_sol) = solution_initiale(number_trucks,listOfClients,data)
        best_sol=curr_sol
        best_cost=curr_cost
        for i in range(nb_iter):
            (curr_cost,best_curr_neigh) = best_voisinage(number_trucks,data,curr_sol,tabou,best_cost,listOfClients)
            if(curr_cost==0):
                break
            if(len(tabou)>=max_tabou_size):
                tabou.pop(0)
            tabou.append((best_curr_neigh,curr_sol))
            curr_sol=best_curr_neigh
            if(curr_cost<best_cost):
                best_cost=curr_cost
                best_sol=curr_sol
        return best_sol,best_cost
    
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
        #prendre un solution aléatoirement dans le pool
        for a in self.model.schedule.agents:
            if isinstance(a,pool_agent):
                if len(a.pool) ==a.nb_solutions:
                    i = rd.randint(0, a.nb_solutions-1)
                    self.solution = closing_tour(algo_RS(self.model.list_clients, self.model.time_matrix,a.pool[i])[0])
                else:
                    self.solution = closing_tour(algo_RS(self.model.list_clients, self.model.time_matrix)[0])
        
        
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
                        if g_bis > g :
                            g = g_bis
                            res = i
                            
                    if res != -1 :
                        self.pool[i]=solution
                        
    def step(self):
        self.solution()
      
class graphic_agent(Agent):
    
    def __init__(self, solution, list_clients):
        self.solution = solution
        self.list_clients = list_clients        
        
    def draw_graph():
        track=truck_track_constructor(self.solution)
        list_clients = self.list_cients
        y = [0]
        z = [0]
        infos = ["Entrepot"]
        for i in range(1,len(list_clients)):

            X.append(list_clients[i].x)
            Y.append(list_clients[i].y)
            txt = "Client n°"+str(list_clients[i].name)+"\n"+str(list_clients[i].quantity)+" - ["+str(list_clients[i].start)+", "+str(list_clients[i].stop)+"]"
            infos.append(txt)


        n = infos
        X=[]
        Y=[]

        plt.figure(figsize=(20,20))
        plt.grid()
        for i in range(n_trucks):
            X2=[]
            Y2=[]
            for j in track[i]:
                X2.append(list_clients[j].x)
                Y2.append(list_clients[j].y)
        
        X.append(X2)
        Y.append(Y2)
        plt.plot(X2,Y2)
        
        for i, txt in enumerate(n):
    
            plt.annotate(txt, (list_clients[i].x, list_clients[i].y))
            plt.title("Score : "+str(self.solution[1])+" | "+str(n_trucks)+" camions d'une capacité de " + str(truck_capacity) + "\nOrdonnancement : " + str(self.solution[0]))
            plt.savefig(str(self.solution[1])+'_chemin.png', format='png') 
            
        def step():
            draw_graph()
        

#la classe SMA
class SMA_collab(Model):
    """A model for infection spread."""

    def __init__(self, nb_pop, nb_generations, n_truck, truck_capacity, list_clients, time_matrix, n_pool,radius_pool):
        self.nb_pop = nb_pop
        self.nb_generations = nb_generations
        self.n_truck = n_truck
        self.truck_capacity = truck_capacity
        self.list_clients = list_clients
        self.time_matrix = time_matrix
        self.n_pool = n_pool
        
        #self.grid = MultiGrid(width, height, True)
            
        self.datacollector = DataCollector(          
            agent_reporters={"State": "state"})

        
        #l'ordonnanceur du modele (instance de RandomActivation)
        #tester SimultaneousActivation (qui permet d'activer tous les agents en mêê temps)
        self.schedule = BaseScheduler(self)

        a = gen_agent(nb_pop, nb_generations, n_trucks, truck_capacity,list_clients)
        self.schedule.add(a)
        
        b = tab_agent(...)
        self.schedule.add(b)
        
        c = rs_agent(...)
        self.schedule.add(c)
        
        # Gestion du pool
        d = pool_agent(4,self,n_pool,radius_pool)
        self.schedule.add(d)
        
        
        # Gestion des courbes
        e1 = graphic_agent(a.best_solution, self.list_clients)
        e2 = graphic_agent(b.solution, self.list_clients)
        e3 = graphic_agent(c.solution, self.list_clients)
        #e = graphic_agent()
        self.schedule.add(e1)
        self.schedule.add(e2)
        self.schedule.add(e3)
        
    def step(self):
        #passage de l'instant t à l'instant (t+1)
        self.schedule.step()
        


        
