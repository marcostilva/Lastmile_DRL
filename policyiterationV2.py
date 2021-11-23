# -*- coding: utf-8 -*-

#Code for DRL policy iteration algorithm
#In V2 following changes were made
## change calls to julia function: just one include and many evals

# Set up environment
import argparse
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import random
import datetime
from time import time
from scipy.stats import norm
from math import ceil, floor

#import torch
#import utils
#from components import Generator, Discriminator
#from torch import nn
#import statsmodels.api as sm
#from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.stattools import acf, pacf
#from statsmodels.tsa.seasonal import seasonal_decompose

print("Moment 1: imported standard Python libs")
# import the necessary packages
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.metrics import classification_report
import csv
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint

print("Moment 2: imported ML Python libs")

# Fixing random seeds
#torch.manual_seed(1368)
rs = np.random.RandomState(1368)
YELLOW_TEXT = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'

import julia
jl = julia.Julia(compiled_modules=False)
from julia import Main
x = Main.include("juliacode.jl")

print("Moment 3: imported Julia code")

#Define algorithms parameters to be tracked and reported




#Define Auxiliary Functions
#Reward Function
#Hard Coded Instance Data

def calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode):
    episodecost=0
    laststop = 0 #depot
    cap = 0
    time = 0
    for i in range(len(DESTC)): # ordering positions
      if scenario[episode][DESTC.index(firststage[i])] == 1 and scenario[episode][len(DESTC) +DESTC.index(firststage[i])] == 0:
        if time + A[laststop, firststage[i]] + A[firststage[i], 0] <= DURV:
          episodecost += REWV*A[laststop,firststage[i]]
          time += A[laststop,firststage[i]]
          laststop = firststage[i]
          cap += 1
          if i == len(DESTC)-1:
            episodecost += REWV*A[firststage[i], 0]
          elif cap == CAPV:
            episodecost += REWV*A[laststop, 0]
            laststop = 0
            cap = 0
            time = 0
        else :
          if i == len(DESTC)-1: # assume 2*time from depot to i ≤ D always
            episodecost += REWV*A[laststop, 0] + REWV*A[0,firststage[i]] + REWV*A[firststage[i], 0]
          else :
            episodecost += REWV*A[laststop, 0] + REWV*A[0, firststage[i]]
            time = A[0, firststage[i]]
            laststop = firststage[i]
            cap = 1
      elif scenario[episode][DESTC.index(firststage[i])] == 1 and scenario[episode][len(DESTC) +DESTC.index(firststage[i])] == 1:
        episodecost += PRICEOD[DESTC.index(firststage[i])]
        if i == len(DESTC)-1 and cap !=0:
          episodecost += REWV*A[laststop, 0]
      elif i == len(DESTC)-1 and cap !=0:
        episodecost += REWV*A[laststop, 0]
      
    return episodecost
    input()
 


def calculate_reward2heuristic(A,DURV,REWV,CAPV,DESTC,firststage,scenario,episode):
    
    laststop = 0 #depot
    cap = 0
    time = 0
    continuo = True
    bypass = False
    episodecost = 0
    i = -1
    while continuo:
      i +=1
      if scenario[episode][DESTC.index(firststage[i])] == 1 and (scenario[episode][len(DESTC) +DESTC.index(firststage[i])] == 0 or bypass):
        bypass = False
        if time + A[laststop, firststage[i]] + A[firststage[i], 0] <= DURV :
          episodecost += REWV*A[laststop,firststage[i]]
          time += A[laststop,firststage[i]]
          laststop = firststage[i]
          cap += 1
          if i == len(DESTC)-1:
            episodecost += REWV*A[firststage[i], 0]
            continuo = False
          elif cap == CAPV:
            episodecost += REWV*A[laststop, 0]
            laststop = 0
            cap = 0
            time = 0
        else :
          if i == len(DESTC)-1: # assume 2*time from depot to i <= timelimit always
            episodecost += REWV*A[laststop, 0] + REWV*A[0, firststage[i]] + REWV*[firststage[i], 0]
            continuo = False
          else :
            episodecost += REWV*A[laststop, 0] + REWV*A[0, firststage[i]]
            time = A[0, firststage[i]] 
            laststop = firststage[i]
            cap = 1
      elif scenario[episode][DESTC.index(firststage[i])] == 1 and scenario[episode][len(DESTC) +DESTC.index(firststage[i])] == 1:
        # find next customer available
        j = i+1
        while j <= len(DESTC)-1 and scenario[episode][DESTC.index(firststage[j])] == 0 :
          j += 1
        if j <= len(DESTC)-1 and PRICEOD[DESTC.index(firststage[i])] <= REWV*A[laststop, firststage[i]] + REWV*A[firststage[i],firststage[j]]:
          episodecost += PRICEOD[DESTC.index(firststage[i])]
          i = j-1
        elif j > len(DESTC)-1 and PRICEOD[DESTC.index(firststage[i])] <= REWV*A[laststop, firststage[i]] + REWV*A[firststage[i], 0]:
          continuo = False
          episodecost += PRICEOD[DESTC.index(firststage[i])]
          if cap != 0: 
            episodecost += REWV*A[laststop, 0]
        elif j <= len(DESTC)-1 and PRICEOD[DESTC.index(firststage[i])] > REWV*A[laststop, firststage[i]] + REWV*A[firststage[i],firststage[j]]:
          bypass = True
          i -= 1
        elif j > len(DESTC)-1 and PRICEOD[DESTC.index(firststage[i])] > REWV*A[laststop, firststage[i]] + REWV*A[firststage[i], 0]:
          bypass = True
          i -= 1
      else :
        if i == len(DESTC)-1 and cap !=0:
          episodecost += REWV*A[laststop, 0]
        if i == len(DESTC)-1:
          continuo = False
    return episodecost
    

def getdata(inst_id):
    #Define size of grid to define Graph
    grid= 10  #will be a 10 x 10 euclidean grid to include depot, customers and ODs

    #Define graph G(V,A)
    V = list(range(grid * grid)) #Depot is vertice/node 0 locate in grid position (0,0)
    A= np.zeros((grid * grid,grid * grid))
    for i in V:                    #i is a vertice/node within the grid
      for j in V:    #j is a vertice/node within the grid 
        A[i,j] = math.sqrt( (i//grid - j//grid)**2 + (i%grid - j%grid)**2)     #building complete graph
  
    #Create Instances Variables
    DESTC = []
    PROBC= []
    REWV = 0
    CAPV = 0
    DURV = 1000000000000
    REWOD= []
    PROBOD= []
  
    if inst_id == "5.1":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [10,55,78,91,99]
      PROBC = [0.3,0.1,0.05,0.1,0.1]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.1,0.05,0.1,0.1]
    elif inst_id == "3.1":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,95]
      PROBC = [0.3333,0.3333,0.3333]
      REWV = 2
      CAPV = 1  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3333,0.3333,0.3333]
    elif inst_id == "4.1":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,95,88]
      PROBC = [0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 1  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3]
    elif inst_id == "5C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "6A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,88,95]
      PROBC = [0.1,0.2,0.3,0.5,0.8,0.95]

      #Define OD Destination nodes and marginal probabilities and fixed reward per distance traveled
      PROBOD = [0.1,0.2,0.3,0.5,0.8,0.95]    #For decision dependent solutioning. two levels of probability and reward
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
    elif inst_id == "6B":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,88,95]
      PROBC = [0.95,0.8,0.5,0.3,0.2,0.1]
      #Define OD Destination nodes and marginal probabilities and fixed reward per distance traveled
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.95,0.8,0.5,0.3,0.2,0.1]
    elif inst_id == "6C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3]
      #Define OD Destination nodes and marginal probabilities and fixed reward per distance traveled
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "6D":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,95,88]
      PROBC = [0.1,0.3,0.6,0.4,0.5,0.7]
      #Define OD Destination nodes and marginal probabilities and fixed reward per distance traveled
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.95,0.8,0.5,0.3,0.2,0.1]
    elif inst_id == "7A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,66,95,88]
      PROBC = [0.1,0.2,0.3,0.5,0.7,0.95,0.8]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.1,0.2,0.3,0.5,0.7,0.95,0.8]
    elif inst_id == "7C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,66,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "9A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,58,66,71,95,88]
      PROBC = [0.1,0.2,0.3,0.5,0.5,0.6,0.7,0.95,0.8]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.1,0.2,0.3,0.5,0.5,0.6,0.7,0.95,0.8]
    elif inst_id == "9B":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,58,66,71,95,88]
      PROBC = [0.95,0.8,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.95,0.8,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
    elif inst_id == "9C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,58,66,71,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "9D":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,58,66,71,95,88]
      PROBC = [0.1,0.3,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.3,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
    elif inst_id == "9F":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,44,55,58,66,71,95,88]
      PROBC = [0.3,0.3,0.3,0.8,0.0,0.8,0.3,0.3,0.8]
      REWV = 2
      CAPV = 9  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.3,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
    elif inst_id == "10A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,88,21,44,55,58,66,71,95,39]
      PROBC = [0.1,0.5,0.3,0.3,0.5,0.5,0.6,0.2,0.95,0.9]
      #PROBC = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.95,0.9]
      REWV = 2
      CAPV = 10  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.2,0.3,0.3,0.5,0.5,0.6,0.7,0.95,0.2]
      #PROBOD=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    elif inst_id == "10ATRIPLE":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,88,21,44,55,58,66,71,95,39]
      #PROBC = [0.9,0.9,0.3,0.3,0.5,0.5,0.6,0.7,0.95,0.9]
      PROBC = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.95,0.9]
      REWV = 2
      CAPV = 10  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      #PROBOD=[0.1,0.2,0.3,0.3,0.5,0.5,0.6,0.7,0.95,0.2]
      PROBOD=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    elif inst_id == "10B":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,55,58,66,71,95,88]
      PROBC = [0.95,0.8,0.8,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.95,0.8,0.8,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
    elif inst_id == "10C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,55,58,66,71,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "10D":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,55,58,66,71,95,88]
      PROBC = [0.1,0.3,0.3,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
      REWV = 2
      CAPV = 3  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.3,0.3,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
    elif inst_id == "11A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,49,55,58,66,71,95,88]
      PROBC = [0.1,0.2,0.3,0.3,0.4,0.5,0.5,0.6,0.7,0.95,0.8]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.2,0.3,0.3,0.4,0.5,0.5,0.6,0.7,0.95,0.8]
    elif inst_id == "11B":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,49,55,58,66,71,95,88]
      PROBC = [0.95,0.8,0.8,0.7,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.95,0.8,0.8,0.7,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
    elif inst_id == "11C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,49,55,58,66,71,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "11D":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,39,44,49,55,58,66,71,95,88]
      PROBC = [0.1,0.3,0.3,0.2,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.3,0.3,0.2,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
    elif inst_id == "12A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,34,39,44,49,55,58,66,71,95,88]
      PROBC = [0.1,0.2,0.25,0.3,0.3,0.4,0.5,0.5,0.6,0.7,0.95,0.8]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.2,0.25,0.3,0.3,0.4,0.5,0.5,0.6,0.7,0.95,0.8]
    elif inst_id == "12B":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,34,39,44,49,55,58,66,71,95,88]
      PROBC = [0.95,0.8,0.8,0.8,0.7,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.95,0.8,0.8,0.8,0.7,0.7,0.3,0.5,0.6,0.5,0.1,0.2]
    elif inst_id == "12C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,34,39,44,49,55,58,66,71,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "12D":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,34,39,44,49,55,58,66,71,95,88]
      PROBC = [0.1,0.3,0.5,0.3,0.2,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
      REWV = 2
      CAPV = 2  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.3,0.5,0.3,0.2,0.6,0.4,0.1,0.4,0.2,0.5,0.7]
    elif inst_id == "14A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [11,21,28,30,39,41,49,55,63,66,71,93,95,88]
      PROBC = [0.1,0.2,0.2,0.25,0.3,0.3,0.4,0.5,0.55,0.6,0.75,0.8,0.95,0.8]
      REWV = 2
      CAPV = 4  #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.2,0.2,0.25,0.3,0.3,0.4,0.5,0.55,0.6,0.75,0.8,0.95,0.8]
    elif inst_id == "20A":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [3,7,11,21,25,28,30,39,41,44,49,55,58,63,66,71,81,93,95,88]
      PROBC = [0.1,0.1,0.1,0.2,0.2,0.2,0.25,0.3,0.3,0.3,0.4,0.5,0.5,0.55,0.6,0.7,0.75,0.8,0.95,0.8]
      REWV = 2
      CAPV = 10 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD=[0.1,0.1,0.1,0.2,0.2,0.2,0.25,0.3,0.3,0.3,0.4,0.5,0.5,0.55,0.6,0.7,0.75,0.8,0.95,0.8]
    elif inst_id == "20C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [3,7,11,21,25,28,30,39,41,44,49,55,58,63,66,71,81,93,95,88]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 10 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "30C":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [3,7,11,24,21,25,28,33,30,39,41,44,49,55,46,58,53,63,66,71,81,93,95,88,91,23,27,29,47,15]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 10 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
    elif inst_id == "30CDOUBLE":
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = [3,7,11,24,21,25,28,33,30,39,41,44,49,55,46,58,53,63,66,71,81,93,95,88,91,23,27,29,47,15]
      PROBC = [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]
      REWV = 2
      CAPV = 10 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    elif inst_id == "70A":
      #Define size of grid to define Graph
      grid= 100  #will be a 10 x 10 euclidean grid to include depot, customers and ODs
      #Define graph G(V,A)
      V = list(range(grid * grid)) #Depot is vertice/node 0 locate in grid position (0,0)
      A= np.zeros((grid * grid,grid * grid))
      for i in V:                    #i is a vertice/node within the grid
        for j in V:    #j is a vertice/node within the grid 
          A[i,j] = math.sqrt( (i//grid - j//grid)**2 + (i%grid - j%grid)**2)     #building complete graph
      #Define customers Destination nodes, marginal probabilities, capacities 
      DESTC = random.sample(list(range(1,grid*grid)),70)
      DESTC = [int(round(x)) for x in DESTC]
      print(DESTC)
      PROBC = [random.choice(np.arange(0.1,0.6,0.1)) for _ in range(70) ]
      print(PROBC)
      REWV = 2
      CAPV = 10 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [random.choice(np.arange(0.1,0.4,0.1)) for _ in range(70) ]
      print(PROBOD)
      input()
    else:
      print("NO INSTANCES FOUND")


    return V,A,DESTC,PROBC,REWV,CAPV,DURV,PROBOD,REWOD



print("Main Program started")

f = open('/home/marco/Dropbox/LASTMILE/APPL/JULIA/RL/result/RESULTDRL.csv', 'a', encoding='UTF8', newline='')

# create the csv writer
writer = csv.writer(f)

###############################################################################
#Define variables to be reported to control performance of different algorithms
RESULT_SIMU_REW1 = 0
RESULT_SIMU_REW2E = 0
RESULT_SIMU_REW2H = 0
RESULT_SIMU_DRO = 0
RESULT_SIMU_REOPT = 0

SOL_FSTAGE_REW1 = 0
SOL_FSTAGE_REW2E = 0
SOL_FSTAGE_REW2H = 0
SOL_FSTAGE_DRO = 0

SOLVALUE_REW1 = 0
SOLVALUE_REW2E = 0
SOLVALUE_REW2H = 0
SOLVALUE_DRO = 0

TIME_DRL_REW1 = 0
TIME_DRL_REW2E = 0
TIME_DRL_REW2H = 0
TIME_DRO = 0

TOTAL_OD_DECISIONS = 0
TOTAL_OD_ALLOCATIONS = 0

ALGOS = ["DRL_REW1","DRL_REW2E","DRL_REW2H","DRO","REPOT"]

ALGO="DRL_REW1"

#size of instance
#key parameters of NN (See later below)
  ##number of iterations
  ##epsilon
  ##learning rate
  ##number of nodes
  ##epochs
  ##batchsize
  ##number of scenarios to train 
  ##number of scenarios to simulate
#instance id see below


header = [datetime.datetime.now(),'Instance', 'Algorithm','# Iterations','Epsilon','Learning rate','# Nodes','EPOCHS','BATCH','# Scen Train','Sol Value REW1','Time REW1','Sol Value REW2E','Time REW2E', 'Sol Value REW2H','Time REW2H','Sol Value DRO','Time DRO','Result Simu REW1','Result Simu REW2E','Result Simu REW2H','Result Simu DRO','Result Simu REOPT', 'TOTAL_OD_DECISIONS','TOTAL_OD_ALLOCATIONS']

writer.writerow(header)
###############################################################################

#Read Instance Data
inst_id ="10A"
V,A,DESTC,PROBC,REWV,CAPV,DURV,PROBOD,REWOD= getdata(inst_id)

#Calculate minimum price to pay for ODs
PRICEOD = [float('inf') for _ in range(len(DESTC))]
DESTC1=DESTC + [0]
for i in DESTC:
  for j in DESTC1:
    for r in DESTC1:
      if (j !=r and j != i and r != i) or (j==0 and r==0):
        if PRICEOD[DESTC.index(i)] > REWV*(A[j,i] + A[i,r] - A[j,r]):
           PRICEOD[DESTC.index(i)] = REWV*(A[j,i] + A[i,r]- A[j,r])
           if PRICEOD[DESTC.index(i)] <= 0.001:  
             PRICEOD[DESTC.index(i)] =0.1 
#PRICEOD  = [3*PRICEOD[i] for i in range(len(PRICEOD))]

Main.V=V
Main.A=A
Main.DESTC=DESTC
Main.PROBC=PROBC
Main.REWV=REWV
Main.CAPV=CAPV
Main.DURV=DURV
Main.PROBOD=PROBOD
Main.REWOD=REWOD
Main.PRICEOD = PRICEOD

print("Instance ", inst_id,"  has been loaded")

#Prepare two groups of scenarios (Evaluation and Simulation)
#Create vectors for training scenarios-policy evaluation  + simulation scenarios -compare different algorithm
LAST_SCENARIO_TO_GENERATE= 50000 #1 EPISODE CONTAINS 4 SCENARIOS (HARDCODED)
LAST_SCENARIO_TO_TRAIN= 49000
FIRST_SCENARIO_TO_SIMULATE= 49000
LAST_SCENARIO_TO_SIMULATE= 50000

scenario = []

for numsce in range(1,LAST_SCENARIO_TO_GENERATE):
  #Will store 4 samples (4 stages in time horizon, per day)
  SAMPLETEST = np.zeros(2*len(DESTC)) #1
  ACCUMTEST = np.zeros(2*len(DESTC))
  x2 = np.random.randint(1,11,2*len(DESTC))
  for i in range(len(DESTC)):
    if x2[i] <= PROBC[i]*10:
      SAMPLETEST[i]=1
  for i in range(len(DESTC),2*len(DESTC)):
    if SAMPLETEST[i- len(DESTC)]== 1 and x2[i] <= PROBOD[i- len(DESTC)]*10:
      SAMPLETEST[i]=1
  scenario.append(SAMPLETEST)
  ACCUMTEST += SAMPLETEST

  SAMPLETEST = np.zeros(2*len(DESTC)) #2
  x2 = np.random.randint(1,11,2*len(DESTC))
  for i in range(len(DESTC)):
    if ACCUMTEST[i]==0 and x2[i] <= PROBC[i]*10:
      SAMPLETEST[i]=1
  for i in range(len(DESTC),2*len(DESTC)):
    if SAMPLETEST[i- len(DESTC)]== 1 and x2[i] <= PROBOD[i- len(DESTC)]*10:
      SAMPLETEST[i]=1
  scenario.append(SAMPLETEST)
  ACCUMTEST += SAMPLETEST

  SAMPLETEST = np.zeros(2*len(DESTC)) #3
  x2 = np.random.randint(1,11,2*len(DESTC))
  for i in range(len(DESTC)):
    if ACCUMTEST[i]==0 and x2[i] <= PROBC[i]*10:
      SAMPLETEST[i]=1
  for i in range(len(DESTC),2*len(DESTC)):
    if SAMPLETEST[i- len(DESTC)]== 1 and x2[i] <= PROBOD[i- len(DESTC)]*10:
      SAMPLETEST[i]=1
  scenario.append(SAMPLETEST)
  ACCUMTEST += SAMPLETEST

  SAMPLETEST = np.zeros(2*len(DESTC)) #4
  x2 = np.random.randint(1,11,2*len(DESTC))
  for i in range(len(DESTC)):
    if ACCUMTEST[i]==0 and x2[i] <= PROBC[i]*10:
      SAMPLETEST[i]=1
  for i in range(len(DESTC),2*len(DESTC)):
    if SAMPLETEST[i- len(DESTC)]== 1 and x2[i] <= PROBOD[i- len(DESTC)]*10:
      SAMPLETEST[i]=1
  scenario.append(SAMPLETEST)
  ACCUMTEST += SAMPLETEST

print("All Scenarios for Policy Evaluation and Simulation Generated...")



#read(STDIN,Char)
################################################


#Solve the RL algorithm

#Define Key parameters
epsilon = 0.6 #must vary in steps of 5%
NUMBERNODES = 32
Main.NUMBERNODES = NUMBERNODES
learning_rate=0.000001
MAXITER = 15
epoch=100
batch=100

#run_DRL(instance,ALGO,NUMBERNODES,epsilon,learning_rate,MAXITER)



def run_DRL(DESTC,CAPV,REWV,ALGO,NUMBERNODES,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN) :
    #Initialize policy (z0 and epsilon)
    bestZ = np.random.permutation(list(range(1,len(DESTC)+1)))
    bestZ = [int(round(x)) for x in bestZ]
     
    #Initialize Neural Networks

    model = Sequential()
    model.add(Dense(NUMBERNODES, input_shape=(len(DESTC),), activation="relu"))
    model.add(Dense(1, activation="linear"))
    sgd = SGD(learning_rate) #Learning rate
    model.compile(loss="mean_squared_error", optimizer=sgd)
    # define the checkpoint
    filepath = "model.h5"


    #Loop Policy Iteration

      #With policy and Scenarios and Reward Function Update Value Function Table

    #Generate value function for all first-stage decisions using first variant recourse
    #By doing that already select the best first-stage decision

    for _  in range(MAXITER):
      print("Initiating Value Function evaluation for algo ", ALGO)
      episode =0
      Value_function_zeta = []
      Value_function = []
      firststage = [0 for _ in range(len(DESTC))]
      while episode < LAST_SCENARIO_TO_TRAIN:
        zetatest = bestZ 
        if random.randint(1,20) <= epsilon*10*2:
          zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
          zetatest = [int(round(x)) for x in zetatest]
        for i in range(len(DESTC)):
          firststage[zetatest[i]-1]= DESTC[i]
        episodecost = 0
        for _ in range(4):
          if ALGO == "DRL_REW1" :
                    episodecost+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode)
          elif ALGO == "DRL_REW2H" :
                    episodecost+=calculate_reward2heuristic(A,DURV,REWV,CAPV,DESTC,firststage,scenario,episode)
          elif ALGO == "DRL_REW2E" :
            #calculate reward 2 exact for this scenario 
            Main.scenario = scenario[episode]
            Main.firststage = firststage
            RES1,RES3,RES2 = jl.eval("rewardvariant2(V,A,REWV,PRICEOD,DESTC,scenario,firststage)") 
            episodecost +=RES2
            print(episode)
          else :
            print("No ALGO to run")
            input()
          episode += 1
        #print(zetatest," , ",episodecost)
        #input()
        Value_function_zeta.append(zetatest)
        Value_function.append(episodecost)

      print("Value Function table mount has ended")

      #Use Value Function table to incremmentally train Neural network
      # train the model using SGD
      print("[INFO] training network...")
      checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
      callbacks_list = [checkpoint]
      H = model.fit(Value_function_zeta, Value_function,epochs=epoch, batch_size=batch,   callbacks=callbacks_list,verbose=2)



      #Use NN parameters to formulate MIP for Policy Improvement
      Main.first_layer_weights = model.layers[0].get_weights()[0]
      first_layer_weights = model.layers[0].get_weights()[0]
      Main.first_layer_biases  = model.layers[0].get_weights()[1]
      first_layer_biases  = model.layers[0].get_weights()[1]
      Main.second_layer_weights = model.layers[1].get_weights()[0]
      Main.second_layer_biases  = model.layers[1].get_weights()[1]

      print("End of NN training")

      #for _ in range(5):
      #  zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
      #  zetatest = [int(round(x)) for x in zetatest]
      #  print(zetatest)
      #  input()   
      #  print(zetatest," ,", model.predict([zetatest]))
      #  input()
    

      #Solve MIP and get new policy (z and epsilon)
      #Do this using Julia (pass NN parameters and Instance Data, get z )

      bestZ, objZ, termination_status = jl.eval("optbestz(DESTC,NUMBERNODES)")
      bestZ = [int(round(x)) for x in bestZ]
      print(bestZ," ,",termination_status," , ",objZ) 
      #input()
    return bestZ, objZ


#run DRL 
start_time = time()
SOL_FSTAGE_REW1,SOLVALUE_REW1 =run_DRL(DESTC,CAPV,REWV,'DRL_REW1',NUMBERNODES,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
TIME_DRL_REW1= time()-start_time

start_time = time()
SOL_FSTAGE_REW2H,SOLVALUE_REW2H =run_DRL(DESTC,CAPV,REWV,'DRL_REW2H',NUMBERNODES,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
TIME_DRL_REW2H= time()-start_time
#SOL_FSTAGE_REW2H=SOL_FSTAGE_REW1

start_time = time()
SOL_FSTAGE_REW2E=SOL_FSTAGE_REW2H
#SOL_FSTAGE_REW2E,SOLVALUE_REW2E =run_DRL(DESTC,CAPV,REWV,'DRL_REW2E',NUMBERNODES,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
TIME_DRL_REW2E= time()-start_time

#Baseline with DRO
start_time = time()
print("Running with DRO baseline if instance size <= 18")
#if len(DESTC) <= 18:
  #Main.inst_idf = inst_id
  #SOL_FSTAGE_DRO,SOLVALUE_DRO = jl.eval("dro(V,A,DESTC,PROBC,PRICEOD,REWV,CAPV,PROBOD,REWOD)")
  #print("DRO ",SOL_FSTAGE_DRO,", ",SOLVALUE_DRO)
  #input()
SOL_FSTAGE_DRO=SOL_FSTAGE_REW1
SOLVALUE_DRO = 0
TIME_DRO= time()-start_time

 
#Initiate simulation comparing best V value function with reoptimization


print("Simulating Value Funtion with REWARD VARIANT 1")
bestZ = SOL_FSTAGE_REW1
print(bestZ)
bestZ = [int(round(x)) for x in bestZ]
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[bestZ[i]-1]= DESTC[i]

episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REW1 = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    #calculate reward 1 for this scenario
    RESULT_SIMU_REW1+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode)
    episode+=1
  

print("Simulating Value Funtion  for REWARD VARIANT 1", RESULT_SIMU_REW1)
#input()

print("Simulating Value Funtion with random solution with REWARD VARIANT 1")

zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
print(zetatest)
zetatest = [int(round(x)) for x in zetatest]
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[zetatest[i]-1]= DESTC[i]
episode =FIRST_SCENARIO_TO_SIMULATE
episodetotalreturn = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    #calculate reward 1 for this scenario
    episodetotalreturn+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode)
    episode +=1
  
print("Simulating Value Funtion  with random solution REWARD VARIANT 1", episodetotalreturn)
#input()


print("Simulating Value Funtion with Second random solution with REWARD VARIANT 1")

zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
print(zetatest)
zetatest = [int(round(x)) for x in zetatest]
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[zetatest[i]-1]= DESTC[i]
episode =FIRST_SCENARIO_TO_SIMULATE
episodetotalreturn = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    #calculate reward 1 for this scenario
    episodetotalreturn+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode)
    episode +=1
print("Simulating Value Funtion  with Second random solution REWARD VARIANT 1", episodetotalreturn)
#input()


print("Simulating Value Funtion with DRO")
bestZ = SOL_FSTAGE_DRO
print(bestZ)

bestZ = [int(round(x)) for x in bestZ]

firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[bestZ[i]-1]= DESTC[i]

episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_DRO = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    #calculate reward 1 for this scenario
    RESULT_SIMU_DRO+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,scenario,firststage,episode)
    episode+=1
  

print("Simulating Value Funtion  with DRO", RESULT_SIMU_DRO)
#input()


print("Simulation using result from best Value Function with REWARD VARIANT 2 EXACT")
bestZ = SOL_FSTAGE_REW2E
print(bestZ)
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[bestZ[i]-1]= DESTC[i]
episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REW2E = 0
TOTAL_OD_DECISIONS = 0
TOTAL_OD_ALLOCATIONS = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
    for _ in range(4):
      if episode%100==0:
        print(episode," ,")
      #calculate reward 2 exact for this scenario 
      Main.scenario = scenario[episode]
      Main.firststage = firststage
      RES1,RES3,RES2= jl.eval("rewardvariant2(V,A,REWV,PRICEOD,DESTC,scenario,firststage)")
      RESULT_SIMU_REW2E+= RES2
      TOTAL_OD_DECISIONS += sum(scenario[episode][len(DESTC):2*len(DESTC)])
      TOTAL_OD_ALLOCATIONS += RES1
      episode +=1
      
print("Simulation using result from best Value Function with REWARD VARIANT 2 EXACT ", RESULT_SIMU_REW2E," ,",TOTAL_OD_ALLOCATIONS," ,",TOTAL_OD_DECISIONS)
#input()

print("Simulating Value Funtion with REWARD VARIANT 2 HEUR")
bestZ = SOL_FSTAGE_REW2H
print(bestZ)
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[bestZ[i]-1]= DESTC[i]

episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REW2H = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%10==0:
      print(episode," ,")
    #calculate reward 2 heur for this scenario 
    RESULT_SIMU_REW2H+=calculate_reward2heuristic(A,DURV,REWV,CAPV,DESTC,firststage,scenario,episode)
    episode +=1
print("Simulation using result from best Value Function with REWARD VARIANT 2 HEUR ", RESULT_SIMU_REW2H)
#input()

print("Simulating Value Funtion with REWARD VARIANT 2 HEUR using REW1 Solution")
bestZ = SOL_FSTAGE_REW1
print(bestZ)
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[bestZ[i]-1]= DESTC[i]

episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REW2H = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%10==0:
      print(episode," ,")
    #calculate reward 2 heur for this scenario 
    RESULT_SIMU_REW2H+=calculate_reward2heuristic(A,DURV,REWV,CAPV,DESTC,firststage,scenario,episode)
    episode +=1
print("Simulation using result from best Value Function with REWARD VARIANT 2 HEUR  using REW1 Solution ", RESULT_SIMU_REW2H)
#input()


print("Simulating Reoptimization: considers Variant 1 type reward")
episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REOPT = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  episodecost = 0
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    Main.scenario = scenario[episode]
    RESULT_SIMU_REOPT += jl.eval("reoptimization(V,A,REWV,PRICEOD,DESTC,scenario)")
    episode +=1
  RESULT_SIMU_REOPT+=episodecost
  print(episode," : ",RESULT_SIMU_REOPT)
print("Simulation using result from Reoptimization ", RESULT_SIMU_REOPT)
#input()

data = [' ',inst_id,ALGO,MAXITER,epsilon,learning_rate,NUMBERNODES,epoch,batch,LAST_SCENARIO_TO_TRAIN,
SOLVALUE_REW1,TIME_DRL_REW1,SOLVALUE_REW2E,TIME_DRL_REW2E, SOLVALUE_REW2H,TIME_DRL_REW2H,SOLVALUE_DRO,TIME_DRO,RESULT_SIMU_REW1,RESULT_SIMU_REW2E,RESULT_SIMU_REW2H,RESULT_SIMU_DRO,RESULT_SIMU_REOPT, TOTAL_OD_DECISIONS,TOTAL_OD_ALLOCATIONS]

writer.writerow(data)
f.close()

