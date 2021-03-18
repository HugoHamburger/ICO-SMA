# -*- coding: utf-8 -*-
"""

SMA Compétition

"""


from mesa import Agent
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector



    
class gen_agent(Agent):
    
    from evolution_functions import next_gen, init_pop, merge_sort
    from constants import nb_generations, nb_pop, n_trucks, truck_capacity, mutation_rate, list_clients
    from evaluation_functions import truck_track_constructor, track_to_member
    import matplotlib.pyplot as plt
    
    
    def __init__(self, n_truck, truck_capacity, list_clients):
        self.n_truck = n_truck
        self.truck_capacity = truck_capacity
        self.list_clients = list_clients
        self.time_matrix = time_matrix     
        self.solution = [] # Meilleure solution à retourner en fin d'éxécution        
       
    
    def step(self):
        
        global gen_agent_last_gen
        
        
        pool = pool_agent.pool
        
        
        try :
            if len(gen_agent_last_gen)>0:
                self.last_pop=True

                #mise en forme de la pool
                pop_pool=[]
                for track in pool :
                    pop_pool.append(track_tp_member(track))
                i=0
                for member in range(pop_pool) :
                    if member[1] < last_gen[0][1] :
                        i+=1
                ratio = i/len(pop_pool)
                
                # Si x % de la pool a un score pluys optimal que le meilleur circuit de la dernière ittération de l'algo génétique
                if i >0:
                    nb_generations = int(nb_generations*(2*ratio))
                    # Eventuellement toucher à l'élite en la diminuant
                
            
                    
        except :
            gen_agent_last_gen = []
            self.last_pop=False
        
        
        # Ajout de la pool à la population
        population = init_pop(nb_pop)
        population[-len(pop_pool):] = pop_pool
        population = merge_sort(population)
              
        
        X=[1]
        Y=[population[0][1]]
        
        for i in range(2,nb_generations+1):
            population = next_gen(population)
            X.append(i)
            Y.append(population[0][1])
        
        self.solution = truck_track_constructor(population[0])
        gen_agent_last_gen = population   
    

        
        
        
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
class SMA_compete(Model):
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
        


        
