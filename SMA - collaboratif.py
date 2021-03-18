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
    
    def __init__(self):
        

        
        
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
        


        