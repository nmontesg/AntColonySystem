# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:29:41 2020

@author: Nieves Montes Gómez

@description: Ant Colony System algorithm to solve the Traveling Salesman problem.
"""

import pandas as pd
import numpy as np
import os
import folium


"""Import data on distances between European cities."""
file = pd.read_csv("distancesEurope.csv", sep=';', index_col = 0)
europe = {}
cities = list(file.columns)
n = len(cities)
for city in cities[:-1]:
  neighbors = cities[cities.index(city)+1:]
  for neighbor in neighbors:
    europe.update({(city, neighbor):file[city][neighbor]})
del city, file, neighbors, neighbor

coordinates = pd.read_csv("coordinates.csv", sep=';', index_col = 0)


"""ACS parameters:"""
alpha = 1. # influence of pheromone values (exploitation)
beta = 2. # influence of edges costs (exploration)
q0 = 0.8 # pseudorandom proportionate rule parameter
phi = 0.04 # pheromone decay coefficient (local pheromone update)
rho = 0.02 # evaporation rate (global pheromone update)
tau0 = 0.01 # initial value of pheromone
max_iter = 20


"""Initialize pheromone values."""
pheromones = {}
for pair in list(europe.keys()):
  pheromones.update({pair: tau0})
del pair


class Ant:
  def __init__(self):
    """Class that represents an ant that follows a path with a certain length."""
    self.path = []
    self.path_length = 0.
  
  def buildNewPath(self):
    """An ant builds a new path according to pseudorandom proportional rule.
    Pheromone values are updated locally. The total distance of the path is
    computed."""
    self.path = [] # erase previous path
    self.path_length = 0.
    self.path.append('Barcelona') # start from BCN
    for _ in range(1,n):
      # select possible next stops
      poss_next = cities[:]
      # delete already visited cities
      for visited in self.path:
        poss_next.remove(visited)
      probs = {}
      # get tau and eta for each next possible option
      for opt in poss_next:
        if (self.path[-1], opt) in europe.keys():
          probs.update({opt : pheromones[(self.path[-1], opt)]**alpha / 
                                         europe[(self.path[-1], opt)]**beta})
        elif (opt, self.path[-1]) in europe.keys():
          probs.update({opt : pheromones[(opt, self.path[-1])]**alpha / 
                                         europe[(opt, self.path[-1])]**beta})
        else:
          raise IndexError('Something went wrong.')
      total = sum(list(probs.values()))
      for k in probs:
        probs[k] = probs[k] / total
      # select next destination: pseudorandom proportional rule
      q = os.urandom(1)[0]/255.
      if q < q0:
        next_index = np.argmax(list(probs.values()))
        self.path.append(list(probs.keys())[next_index])
      else:
        next_city = np.random.choice(list(probs.keys()), 
                                     p = list(probs.values()))
        self.path.append(next_city)
      # local pheromone update and add edge cost to total distance
      if (self.path[-1], self.path[-2]) in pheromones.keys():
        pheromones[(self.path[-1], self.path[-2])] = (1-phi) * pheromones[
            (self.path[-1], self.path[-2])] + phi*tau0
        self.path_length += europe[(self.path[-1], self.path[-2])]
      elif (self.path[-2], self.path[-1]) in pheromones.keys():
        pheromones[(self.path[-2], self.path[-1])] = (1-phi) * pheromones[
            (self.path[-2], self.path[-1])] + phi*tau0
        self.path_length += europe[(self.path[-2], self.path[-1])]
      else:
        raise IndexError('Something went wrong.')
    # after exit the loop: local pheromone and distance update on edge closing the cycle
    if (self.path[-1], 'Barcelona') in pheromones.keys():
      pheromones[(self.path[-1], 'Barcelona')] = (1-phi) * pheromones[
          (self.path[-1], 'Barcelona')] + phi*tau0
      self.path_length += europe[(self.path[-1], 'Barcelona')]
    elif ('Barcelona', self.path[-1]) in pheromones.keys():
      pheromones[('Barcelona', self.path[-1])] = (1-phi) * pheromones[
          ('Barcelona', self.path[-1])] + phi*tau0
      self.path_length += europe[('Barcelona', self.path[-1])]
    else:
      raise IndexError('Something went wrong.')
      
  def mapPath(self):
    """Plot the ant's path on the Europe map."""
    m = folium.Map(location=(48, 10), zoom_start = 5, min_zoom = 4, max_zoom = 10)
    points = []
    for city in self.path:
      points.append((coordinates.loc[city, 'lat'], coordinates.loc[city, 'lon']))
      folium.Marker((coordinates.loc[city, 'lat'], coordinates.loc[city, 'lon']),
                    popup = folium.Popup(city),
                    icon = folium.Icon(color='red', icon='heart')).add_to(m)
    points.append((coordinates.loc[self.path[0], 'lat'], 
                   coordinates.loc[self.path[0], 'lon']))
    folium.PolyLine(points, color = 'red', weight = 5, opacity = 0.75).add_to(m)
    return m
      
  def __str__(self):
    return ('Hi, I\'m a curious ant exploring Europe. I can visit all cities in a ' + 
            str(self.path_length) + ' km tour.')

 
class Colony:
  def __init__(self, colsize=20):
    """Class that represents a colony of ants. colsize is the number of ants.
    Ants are randomly initialized."""
    self.size = colsize
    self.members = [Ant() for _ in range(colsize)]
    self.best = self.members[0]
    for member in self.members:
      member.buildNewPath()
    self.findBest()
  
  def findBest(self):
    """Find the ant in the current iteration that has the shortest path."""
    for member in self.members:
      if member.path_length < self.best.path_length:
        self.best = member
  
  def newPaths(self):
    """Send all ants in the colony in search of a new path."""
    for member in self.members:
      member.buildNewPath()
        
  def globalPheromoneUpdate(self):
    """Global pheromone update. Must be called after best ant has already been found."""
    DeltaBest = 1/self.best.path_length
    # common update to all edges
    for edge in pheromones.keys():
      pheromones[edge] *= (1-rho)
    # update to edges in best path
    for i in range(len(self.best.path) - 1):
      if (self.best.path[i], self.best.path[i+1]) in pheromones.keys():
        pheromones[(self.best.path[i], self.best.path[i+1])] += rho*DeltaBest
      elif (self.best.path[i+1], self.best.path[i]) in pheromones.keys():
        pheromones[(self.best.path[i+1], self.best.path[i])] += rho*DeltaBest
      else:
        raise IndexError('Something went wrong.')
    # edges that closes the cycle of the best path
    if (self.best.path[0], self.best.path[-1]) in pheromones.keys():
      pheromones[(self.best.path[0], self.best.path[-1])] += rho*DeltaBest
    elif (self.best.path[-1], self.best.path[0]) in pheromones.keys():
      pheromones[(self.best.path[-1], self.best.path[0])] += rho*DeltaBest
    else:
      raise IndexError('Something went wrong.')
             
        
"""Ant Colony System"""
import copy
col = Colony()
bestSoFar = copy.deepcopy(col.best)
iters = 1
stop = 50
bestIter = 0

while iters <= stop:
  col.newPaths()
  col.findBest()
  col.globalPheromoneUpdate()
  if col.best.path_length < bestSoFar.path_length:
    bestSoFar = copy.deepcopy(col.best)
    stop += max_iter
    bestIter = iters
  iters += 1

# output result
print('Best solution found during iteration #' + str(bestIter) + '.')
print('This clever ant visits all cities in a ' + str(bestSoFar.path_length) + 
      ' km tour.')
print('Her adventorous journey is:', end = ' ')
for city in bestSoFar.path:
  print(city, end = '-')
print(bestSoFar.path[0])

# export map of route as html
m = bestSoFar.mapPath()
m.save('tour.html')