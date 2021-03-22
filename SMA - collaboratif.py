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
                g+= distance(y,pool[i])
                
                
    def solution(self):
        for a in self.model.schedule.agents:
            if isinstance(a,tab_agent) or isinstance(a,rs_agent) or isinstance(a,gen_agent):
                solution = a.solution
                res = -1
                if len(self.pool)< nb_solutions :
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
        


        
