# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:19:46 2021

@author: Julien
"""
from classes import Client
###############################################################################
######################   Enregistrement des constantes   ######################
###############################################################################

#Équipements        
n_trucks = 4 #Nombre de camions
truck_capacity = 25 #Capacité unitaire pouvant être emmenée par chaque camion

#Paramètres AGs
nb_pop = 200 #taille de notre population
nb_generations = 100 #nombre de générations étudiées par l'AGs
elitism = True #Volonté de ne sélectionner que les meilleurs éléments
best_pop = 40 #Taille de la population élite : les N-meilleurs membres de la population
mutation_rate = 0.3 #probabilité de mutation lors du passage à la génération suivante

#Pénalités
time_penalty = 100 #Pénalité si un client n'est pas livré dans les temps
quantity_penalty = 100 #Pénalité si on ne livre pas intégralement un client
truck_cost = 0 #Coût d'utilisation d'un camion

###############################################################################
########################   Paramètres du problème      ########################
###############################################################################

#On définit d'abord la matrice des temps depuis https://developers.google.com/optimization/routing/vrptw?fbclid=IwAR1Cy40SLDbDJqmIlqQclEQVtuHwwQdPEJ7G0ufurS0fYV-KIemjkwc3gDM
time_matrix = [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
    ]

#On définit les clients à livrer selon la syntaxe suivante : 
#Client(nom du client, coordonnée x dans le plan, coordonnée y dans le plan,
#       quantité à livrer, début de la fenêtre de livraison, fin de la fenêtre de livraison)

list_clients = []
list_clients.append(Client(0,0,0,0,0,1000))
list_clients.append(Client(1,-2,4,1,7,12))
list_clients.append(Client(2,4,4,1,10,15))
list_clients.append(Client(3,-4,3,3,16,18))
list_clients.append(Client(4,-3,3,4,10,13))
list_clients.append(Client(5,1,2,2,0,5))
list_clients.append(Client(6,3,2,4,5,10))
list_clients.append(Client(7,-1,1,8 ,0,4))
list_clients.append(Client(8,2,1,8,5,10))
list_clients.append(Client(9,1,-1,1,0,3))
list_clients.append(Client(10,4,-1,2,10,16))
list_clients.append(Client(11,-3,-2,1,10,15))
list_clients.append(Client(12,-2,-2,2,0,5))
list_clients.append(Client(13,-1,-3,14,5,10))
list_clients.append(Client(14,2,-3,4,7,8))
list_clients.append(Client(15,-4,-4,8,10,15))
list_clients.append(Client(16,3,-4,8,11,15))

#On récupère ici une liste tronquée de nos clients avec les paramètres suivants :
#   quantité à livrer, début de la fenêtre de livraison, fin de la fenêtre de livraison

clients= [[list_clients[i].quantity, list_clients[i].start, list_clients[i].stop] for i in range(len(list_clients))]