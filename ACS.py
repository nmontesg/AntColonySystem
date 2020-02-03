# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:29:41 2020

@author: Nieves Montes Gómez

@description: Ant Colony System algorithm to solve the Travelling Salesman problem.
"""

import pandas as pd
import numpy as np
import os

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


"""ACS parameters:"""
alpha = 1. # influence of pheromone values (exploitation)
beta = 1. # influence of edges costs (exploration)
q0 = 0.5 # pseudorandom proportionate rule parameter
phi = 0.4 # pheromone decay coefficient (local pheromone update)
rho = 0.3 # evaporation rate (global pheromone update)
tau0 = 0.01 # initial value of pheromone
max_iter = 15


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
          probs.update({opt : pheromones[(self.path[-1], opt)]**alpha / europe[(self.path[-1], opt)]**beta})
        elif (opt, self.path[-1]) in europe.keys():
          probs.update({opt : pheromones[(opt, self.path[-1])]**alpha / europe[(opt, self.path[-1])]**beta})
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
        next_city = np.random.choice(list(probs.keys()), p = list(probs.values()))
        self.path.append(next_city)
      # local pheromone update and add edge cost to total distance
      if (self.path[-1], self.path[-2]) in pheromones.keys():
        pheromones[(self.path[-1], self.path[-2])] = (1-phi) * pheromones[(self.path[-1], self.path[-2])] + phi*tau0
        self.path_length += europe[(self.path[-1], self.path[-2])]
      elif (self.path[-2], self.path[-1]) in pheromones.keys():
        pheromones[(self.path[-2], self.path[-1])] = (1-phi) * pheromones[(self.path[-2], self.path[-1])] + phi*tau0
        self.path_length += europe[(self.path[-2], self.path[-1])]
      else:
        raise IndexError('Something went wrong.')
    # after exit the loop: local pheromone and distance update from last to Barcelona
    if (self.path[-1], 'Barcelona') in pheromones.keys():
      pheromones[(self.path[-1], 'Barcelona')] = (1-phi) * pheromones[(self.path[-1], 'Barcelona')] + phi*tau0
      self.path_length += europe[(self.path[-1], 'Barcelona')]
    elif ('Barcelona', self.path[-1]) in pheromones.keys():
      pheromones[('Barcelona', self.path[-1])] = (1-phi) * pheromones[('Barcelona', self.path[-1])] + phi*tau0
      self.path_length += europe[('Barcelona', self.path[-1])]
    else:
      raise IndexError('Something went wrong.')
      
#  def mapPath(self):
#    """Plot the ant's path on the Europe map."""

  
class Colony:
  def __init__(self, colsize=50):
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
    for member in self.members:
      for i in range(len(member.path) - 1):
        if (member.path[i], member.path[i+1]) in pheromones.keys():
          pheromones[(member.path[i], member.path[i+1])] = (1-phi) * pheromones[(member.path[i], member.path[i+1])] + phi*DeltaBest
        elif (member.path[i+1], member.path[i]) in pheromones.keys():
          pheromones[(member.path[i+1], member.path[i])] = (1-phi) * pheromones[(member.path[i+1], member.path[i])] + phi*DeltaBest
        else:
          raise IndexError('Something went wrong.')
      # update pheromones in edge that closes the cycle
      if (member.path[0], member.path[-1]) in pheromones.keys():
        pheromones[(member.path[0], member.path[-1])] = (1-rho) * pheromones[(member.path[0], member.path[-1])] + rho*DeltaBest
      elif (member.path[-1], member.path[0]) in pheromones.keys():
        pheromones[(member.path[-1], member.path[0])] = (1-rho) * pheromones[(member.path[-1], member.path[0])] + rho*DeltaBest
      else:
        raise IndexError('Something went wrong.')
        
        
"""Ant Colony System:"""
import copy
col = Colony()
bestSoFar = copy.deepcopy(col.best)
print(col.best.path_length)
print(bestSoFar.path_length)
iters = 1
stop = max_iter

while iters <= stop:
  col.newPaths()
  col.findBest()
  col.globalPheromoneUpdate()
  if col.best.path_length < bestSoFar.path_length:
    bestSoFar = copy.deepcopy(col.best)
    stop += max_iter
  iters += 1
  print(col.best.path_length)