# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:22:06 2021

@author: Julien
"""

from constants import time_penalty, quantity_penalty, time_matrix, clients, truck_capacity, truck_cost
###############################################################################
##########################   Fonctions d'évaluation  ##########################
###############################################################################


def track_evaluation(truck_track):
    t3=0
    trucks_time = []
    for track in truck_track:
        t=0
        q=truck_capacity
        if len(track) >2 :
            t3+= truck_cost #Si le camion est utilisé, ajout du coût d'utilisation dans le score du circuit
        for i in range(1,len(track)):
            
            # Evaluation fenêtre
            t2= time_matrix[track[i-1]][track[i]]
            if t+t2 <= clients[track[i]][2]: #Si arrivé du livreur avant la fin de la fenêtre
                t+=t2
                t+=+max(0,clients[track[i]][1]-t)  #ajout d'un temps d'attente si le livreur arrive avant le début de la fenêtre
            else :
                t3+= (t+t2 - clients[track[i]][2])*time_penalty #ajout d'une pénalité si une fenêtre n'est pas respectée
                t+=t2
                               
            # Evaluation quantité            
            if q >= clients[track[i]][0]:
                q-=clients[track[i]][0]
            else : 
                t3+=(clients[track[i]][0]-q)*quantity_penalty   #ajout d'une pénalité si une quantité n'est pas respectée
                q=0                
        trucks_time.append(t)
        t3+=t
    return t3,trucks_time
    
                
def truck_track_constructor(member):
    track=member[0]   


    cgt=[0]
    truck_track=[]
    
    for j in range(1,len(track)):
        if track[j] <= 0:           
            cgt.append(j)
            truck_track.append([0] + track[cgt[-2]+1:cgt[-1]] + [0])
            
    cgt.append(j)
    truck_track.append([0] + track[cgt[-2]+1:cgt[-1]+1] + [0])  

    return truck_track

def population_evaluation(member): 
    t3,trucks_time=track_evaluation(truck_track_constructor(member))        
    member[1]=t3
    return member

#Nous avons reprogrammé ci-dessous un tri fusion
def merge(left,right): 
    result = []
    index_left, index_right = 0, 0
    while index_left < len(left) and index_right < len(right):        
        if left[index_left][1] <= right[index_right][1]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
    if left:
        result.extend(left[index_left:])
    if right:
        result.extend(right[index_right:])
    return result
 
def merge_sort(m):
    if len(m) <= 1:
        return m
    middle = len(m) // 2
    left = m[:middle]
    right = m[middle:]
    left = merge_sort(left)
    right = merge_sort(right)
    return list(merge(left, right))

def track_to_member(track):
    i=0
    member=[[],track_evaluation(track)[0]]
    for truck_track in track:
        member[0]= member[0]+[i]+truck_track[1:-1]
        i-=1
    return(member)
    