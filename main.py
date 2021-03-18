# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:18:36 2021

@author: Julien
"""

from evolution_functions import next_gen, init_pop
from constants import nb_generations, nb_pop, n_trucks, truck_capacity, mutation_rate, list_clients
from evaluation_functions import truck_track_constructor

import matplotlib.pyplot as plt


import time
t1 = time.perf_counter()
###############################################################################
#######################   génération de la population  ########################
###############################################################################
population=init_pop(nb_pop)

X=[1]
Y=[population[0][1]]

for i in range(2,nb_generations+1):
    population = next_gen(population)
    X.append(i)
    Y.append(population[0][1])
plt.plot(X,Y)
plt.xlabel('génération')
plt.ylabel("Score")
plt.title("Score : "+str(population[0][1])+" | "+str(n_trucks)+" camions d'une capacité de " + str(truck_capacity)+"\nTaux de mutation : "+str(mutation_rate)+" | Taile de la population : "+str(nb_pop) )    
#    if i%100 == 0: 
#        print(population[0][1])
    
plt.savefig(str(population[0][1])+'_distance.png', format='png')    


track=truck_track_constructor(population[0])


 #on genere une carte représentant notre solution
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
plt.title("Score : "+str(population[0][1])+" | "+str(n_trucks)+" camions d'une capacité de " + str(truck_capacity) + "\nOrdonnancement : " + str(population[0][0]))
plt.savefig(str(population[0][1])+'_chemin.png', format='png') 

t = time.perf_counter() - t1
print(t)