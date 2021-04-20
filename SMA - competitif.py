# -*- coding: utf-8 -*-
"""

SMA Compétition

"""


from mesa import Agent
from mesa import Model
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector



          

class gen_agent(Agent):
    

    
    
    def __init__(self,unique_id,model, nb_pop, nb_generations, n_truck, truck_capacity, list_clients):
        super().__init__(unique_id,model)
        self.nb_pop = nb_pop
        self.nb_generations =  nb_generations
        self.n_trucks = n_trucks
        self.truck_capacity = truck_capacity
        self.list_clients = list_clients
        self.time_matrix = time_matrix     
        self.best_solution = [] # Meilleure solution à retourner en fin d'éxécution        
        self.population = init_pop(nb_pop)
        self.solution = truck_track_constructor(self.population[0])    
        self.best_pop = best_pop
       
    
    def step(self):
      
        global gen_agent_last_gen                

        
        
        for a in self.model.schedule.agents:
           if   isinstance(a, pool_agent): 
               pool = a.pool

        
        # Ajout de la pool à la population
        population = init_pop(nb_pop)
        
        if len(pool) > 1 :
            self.last_pop=True
            # mise en forme de la pool
            pop_pool=[]
            for track in pool :
                pop_pool.append(track_to_member(track))
            i=0
            for member in pop_pool :
                if member[1] < gen_agent_last_gen[0][1] :
                    i+=1
            ratio = i/len(pop_pool)                
            # Si x % de la pool a un score pluys optimal que le meilleur circuit de la dernière ittération de l'algo génétique
            if i >0:
                self.best_pop = max(self.best_pop - 5, 20)
                print(self.best_pop)
                
                constants_update(self.best_pop)
             
            population[-len(pop_pool):] = pop_pool
            population = merge_sort(population)     
                # Eventuellement toucher à l'élite en la diminuant
            
        else :
            gen_agent_last_gen = []
            self.last_pop=False
            
        
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
        
      
        



def distanceTab(C1, C2, listOfClients):
        return(np.sqrt((listOfClients[C1].y-listOfClients[C2].y)**2 + (listOfClients[C1].x - listOfClients[C2].x)**2))
    

      
class graphic_agent(Agent):
    
    def __init__(self,name, model, solution, list_clients):
        super().__init__(name, model)
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
class pool_agent(Agent):
    def __init__(self, name, model, nb_solutions,pool_radius):
        super().__init__(name,model)
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
            if   isinstance(a,gen_agent): # or isinstance(a,tab_agent) or isinstance(a,rs_agent) 
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
        print(self.pool)  
        
        
        
        

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

        a = gen_agent(1,self,nb_pop, nb_generations, n_trucks, truck_capacity,list_clients)
        self.schedule.add(a)
        

        
        # b = tab_agent(2,self)
        # self.schedule.add(b)
        
        # c = rs_agent(3, self, n_trucks, truck_capacity)
        # self.schedule.add(c)
        
        # Gestion du pool
        d = pool_agent(4,self,n_pool,radius_pool)
        self.schedule.add(d)
        
        
        # Gestion des courbes
        e1 = graphic_agent(5,self,a.solution, self.list_clients)
        # e2 = graphic_agent(6,self,b.solution, self.list_clients)
        # e3 = graphic_agent(7,self,c.solution, self.list_clients)
        #e = graphic_agent()
        self.schedule.add(e1)
        # self.schedule.add(e2)
        # self.schedule.add(e3)
        
    def step(self):
        #passage de l'instant t à l'instant (t+1)
        self.schedule.step()
        # print(model.schedule.agents[1].pool)

        


steps=20 

n_truck = 4
n_pool = 10
radius_pool = 10

model = SMA_collab(nb_pop, nb_generations, n_truck, truck_capacity, list_clients, time_matrix, n_pool,radius_pool)

for i in range(steps):
    model.step()       
