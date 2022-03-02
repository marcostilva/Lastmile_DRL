# -*- coding: utf-8 -*-

#Code for DRL policy iteration algorithm


# Set up environment
import argparse
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import datetime
from time import time
from scipy.stats import norm
from math import ceil, floor
import pickle
import gc

print("Moment 1: imported standard Python libs. Now ML Libs...")
# import the necessary packages
import csv
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint

from docplex.mp.solution import SolveSolution
from docplex.mp.model import Model

import sys

from cplex.callbacks import HeuristicCallback, UserCutCallback, LazyConstraintCallback

from docplex.mp.callbacks.cb_mixin import *


print("Moment 2: imported ML and CPLEX Python libs.")

# Fixing random seeds
#torch.manual_seed(1368)
rs = np.random.RandomState(1368)
YELLOW_TEXT = '\033[93m'
ENDC = '\033[0m'
BOLD = '\033[1m'


#Define algorithms parameters to be tracked and reported




#Define Auxiliary Functions
#Reward Function
#Hard Coded Instance Data

class DOHeurCallback(ModelCallbackMixin, HeuristicCallback):
    def __init__(self, env):
        HeuristicCallback.__init__(self, env)
        ModelCallbackMixin.__init__(self)

    @print_called('--> calling my_heuristic callback... #{0}')
    def __call__(self):
        feas = self.get_feasibilities()
        var_indices = [j for j, f in enumerate(feas)] # if f ==self.feasibility_status.feasible]
        #print("set solution0")
        if var_indices:
            #print("set solution1")
            # this shows how to get back to the DOcplex variable from the index
            # but is not necessary for the logic.
            dvars = [self.index_to_var(v) for v in var_indices]
            sol=self.make_solution_from_vars(dvars)
            z_val =[]
            s_val =[]
            z_indices=[]
            for i in range(NUMBERNODES):
              s_val += sol.get_values([v for v in dvars if v.name=='s_'+str(i) ]) 
            #print("set solution2")  
            for i in range(len(DESTC)):
              z_val += sol.get_values([v for v in dvars if v.name=='z_'+str(i) ])
              z_indices += [v.index for v in dvars if v.name=='z_'+str(i)]   
            #Create Vector with Customer delivery order suggested by z_val 
            #print("set solution2")
            z_order = np.zeros(len(DESTC),dtype=int)
            for i in range(len(DESTC)):
              order = 1
              for j in range(len(DESTC)):
                if i != j: 
                  if z_val[j] < z_val[i]:
                    order += 1
              z_order[i]= order
            #Now define heuristic for variables x using z_order
            #print("set solution3")
            x_indices = []
            x_val =[]
            for i in range(len(DESTC)-1):
              for j in range(i+1,len(DESTC)):
                #print(i,j)
                if z_order[i] < z_order[j]:
                  x_val += [1]
                  x_indices += [v.index for v in dvars if v.name=='x_'+str(i)+'_'+str(j)]
                else:
                  x_val += [0]
                  x_indices += [v.index for v in dvars if v.name=='x_'+str(i)+'_'+str(j)]
 
            #print('* rounded vars = [{0}]'.format(', '.join([v.name for v in dvars[:9]])))
            #print("set solution")
            # -- calling set-solution in cplex callback class
            self.set_solution([z_indices+x_indices, list(z_order)+x_val])
        #print("bye bye")
      
            
# Lazy constraint callback to separate subtour elimination constraints.
class DOLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        #self.nb_lazy_cts = 0

  
    @print_called('--> lazy constraint callback called: #{0}')
    def __call__(self):
        # fetch variable values into a solution
        
        #sol = self.make_solution()
        ## for each lazy constraint, check whether it is verified,
        #if not sol.is_empty():
        #  unsats = self.get_cpx_unsatisfied_cts(self.cts, sol, tolerance=1e-6)
        #  if len(unsats) > 0:
        #    for ct, cpx_lhs, sense, cpx_rhs in unsats:
        #      self.add(cpx_lhs, sense, cpx_rhs)
        #      #break
        #    return 
        
         
        # fetch variable values into a solution
        #print("lazy1...")
        feas = self.get_feasibilities()
        var_indices = [j for j, f in enumerate(feas)] 
        z_val =[]
        s_val =[]
        y_val =[]
        x_val =[]
        z_indices=[]
        s_indices=[]
        y_indices=[]
        x_indices=[]
        if var_indices:
          #print("lazy2...")
          # this shows how to get back to the DOcplex variable from the index
          # but is not necessary for the logic.
          dvars = [self.index_to_var(v) for v in var_indices]
          sol=self.make_solution_from_vars(dvars)
          for i in range(NUMBERNODES):
            s_val += sol.get_values([v for v in dvars if v.name=='s_'+str(i)])
            y_val += sol.get_values([v for v in dvars if v.name=='y_'+str(i)])
            s_indices += [v.index for v in dvars if v.name=='s_'+str(i)]
            y_indices += [v.index for v in dvars if v.name=='y_'+str(i)] 
          #print("lazy3...")
          for i in range(len(DESTC)):
            z_val += sol.get_values([v for v in dvars if v.name=='z_'+str(i)])
            z_indices += [v.index for v in dvars if v.name=='z_'+str(i)] 
            x_valrow=[]
            x_indicesrow=[]
            for j in range(len(DESTC)):
              #print("lazy4...",i,",",j)
              x_valrow += sol.get_values([v for v in dvars if v.name=='x_'+str(i)+'_'+str(j)])
              x_indicesrow += [v.index for v in dvars if v.name=='x_'+str(i)+'_'+str(j)] 
            x_indicesrow =  [0.0] * (i+1)+x_indicesrow   
            x_valrow =  [0.0 ]* (i+1)+x_valrow
            x_val.append(x_valrow)  
            x_indices.append(x_indicesrow) 
          #print("lazy5...")      
          for i in range(len(DESTC)):
            for j in range(len(DESTC)):
              for k in range(len(DESTC)):
                #print(i,j,k)
                if i > j and j < k and k < i and -x_val[j][i]+x_val[j][k]+x_val[k][i]>1:
                  #print("1...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(-self.index_to_var(x_indices[j][i])+ self.index_to_var(x_indices[j][k])+ self.index_to_var(x_indices[k][i])<=1)
                  self.add(lhs,sense,rhs)
                  return
                elif i < j and j > k and k < i and x_val[i][j]-x_val[k][j]+x_val[k][i]>1:
                  #print("2...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(self.index_to_var(x_indices[i][j])-self.index_to_var(x_indices[k][j])+self.index_to_var(x_indices[k][i])<=1)
                  self.add(lhs,sense,rhs)
                  return
                elif i > j and j > k and k < i and -x_val[j][i]-x_val[k][j]+ x_val[k][i]>0:
                  #print("3...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(-self.index_to_var(x_indices[j][i])-self.index_to_var(x_indices[k][j])+self.index_to_var(x_indices[k][i])<=0)
                  self.add(lhs,sense,rhs)
                  return
                elif i < j and j < k and k > i and x_val[i][j]+x_val[j][k]-x_val[i][k]>1:
                  #print("4...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(self.index_to_var(x_indices[i][j])+self.index_to_var(x_indices[j][k])-self.index_to_var(x_indices[i][k]) <=1)
                  self.add(lhs,sense,rhs)
                  return
                elif i > j and j < k and k > i and -x_val[j][i]+x_val[j][k]-x_val[i][k]>0:
                  #print("5...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(-self.index_to_var(x_indices[j][i])+self.index_to_var(x_indices[j][k])-self.index_to_var(x_indices[i][k]<=0))
                  self.add(lhs,sense,rhs)
                  return
                elif i < j and j > k and k > i and x_val[i][j]-x_val[k][j]-x_val[i][k]>0:
                  #print("6...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(self.index_to_var(x_indices[i][j])-self.index_to_var(x_indices[k][j])-self.index_to_var(x_indices[i][k]) <=0)
                  self.add(lhs,sense,rhs)
                  return
                elif i > j and j > k and k > i and -x_val[j][i]-x_val[k][j]-x_val[i][k]>-1:
                  #print("7...")
                  lhs,sense,rhs= self.linear_ct_to_cplex(-self.index_to_var(x_indices[j][i])-self.index_to_var(x_indices[k][j])-self.index_to_var(x_indices[i][k])<=-1)
                  self.add(lhs,sense,rhs)
                  return

          #print("8...") 
          for node in range(NUMBERNODES):
            m2 = Model(name='sepprob',log_output=False)
            m2.context.cplex_parameters.timelimit = 600
  
            neta = m2.binary_var_list(len(DESTC), name="n")
            m2.set_objective("min", sum( neta[i]*first_layer_weights[i,node]*(z_val[i]-L[node,i]*(1-s_val[node])) for i in range(len(DESTC)) if first_layer_weights[i,node] != 0) 
      + s_val[node]*(first_layer_biases[node]+ sum((1-neta[i])*U[node,i]*first_layer_weights[i,node] for i in range(len(DESTC)) if first_layer_weights[i,node] == 0) ))
      
            
           
            m2.add_constraints_( (neta[i]<=1 for i in range(len(DESTC))))
            
            msol2=m2.solve()
            if not msol2.is_empty() and msol2.get_objective_value() < y_val[node]:
              neta_val = msol2.get_values([neta[ind] for ind in range(len(DESTC)) ])
              lhs,sense,rhs= self.linear_ct_to_cplex(self.index_to_var(y_indices[node]) <= sum(neta_val[i]*first_layer_weights[i,node]*(self.index_to_var(z_indices[i])-L[node,i]*(1-self.index_to_var(s_indices[node]))) for i in range(len(DESTC)) if first_layer_weights[i,node] != 0) + self.index_to_var(s_indices[node])*((1-neta_val[i])*first_layer_biases[node]+ sum((1-neta_val[i])*U[node,i]*first_layer_weights[i,node] for i in range(len(DESTC)) if first_layer_weights[i,node] != 0)))
              self.add(lhs,sense,rhs)
              break  
            
  


def calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode):
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
    grid= 100 #will be a 10 x 10 euclidean grid to include depot, customers and ODs

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
  
    if inst_id[-1] == '0':
      siz=int(inst_id)
      #Define customers Destination nodes, marginal probabilities, capacities
      DESTC = random.sample(list(range(1,grid*grid)),siz)
      DESTC = [int(round(x)) for x in DESTC]
      #print(DESTC)
      PROBC = [random.choice(np.arange(0.1,0.3,0.1)) for _ in range(siz) ]
      #print(PROBC)
      REWV = 2
      CAPV = siz/3 #demand is given in demands unit, each customer is fixed at one unit
      REWOD= []
      PROBOD= [random.choice(np.arange(0.1,0.3,0.1)) for _ in range(siz) ]
      #print(PROBOD)
      #input()
    else:
      print("NO INSTANCES FOUND")


    return V,A,DESTC,PROBC,REWV,CAPV,DURV,PROBOD,REWOD



print("Main Program started")

f = open('/home/DRL/result/RESULTDRL.csv', 'a', encoding='UTF8', newline='')
# create the csv writer
writer = csv.writer(f)

###############################################################################
#Define variables to be reported to control performance of different algorithms
RESULT_SIMU_REW1 = 0
RESULT_SIMU_REW1_RAND = 0
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


header = [datetime.datetime.now(),'Instance', 'Algorithm','# Iterations','Epsilon','Learning rate','# Nodes','EPOCHS','BATCH','# Scen Train','Sol Value REW1','Time REW1','Sol Value REW2E','Time REW2E', 'Sol Value REW2H','Time REW2H','Sol Value DRO','Time DRO','Result Simu REW1','Result Simu REW1 Random','Result Simu REW2E','Result Simu REW2H','Result Simu DRO','Result Simu REOPT', 'TOTAL_OD_DECISIONS','TOTAL_OD_ALLOCATIONS']

writer.writerow(header)
###############################################################################

#Read Instance Data
print("Read Instance Data")
inst_id ="70"
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

print("Instance ", inst_id,"  has been loaded")

#Prepare two groups of scenarios (Evaluation and Simulation)
#Create vectors for training scenarios-policy evaluation  + simulation scenarios -compare different algorithm
LAST_SCENARIO_TO_GENERATE= 100000 #1 EPISODE CONTAINS 4 SCENARIOS (HARDCODED)
LAST_SCENARIO_TO_TRAIN= 99000
FIRST_SCENARIO_TO_SIMULATE= 99000
LAST_SCENARIO_TO_SIMULATE= 100000

print("Going to Generate Scenarios for training...")
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

  SAMPLETEST = np.zeros(2*len(DESTC))
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
#Main.NUMBERNODES = NUMBERNODES
learning_rate=0.000001
MAXITER = 15
epoch=100
batch=100


def run_DRL(DESTC,CAPV,REWV,ALGO,NUMBERNODES,scenario,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN) :
    #Initialize policy (z0 and epsilon)
    bestZ = np.random.permutation(list(range(1,len(DESTC)+1)))
    bestZ = [int(round(x)) for x in bestZ]
     
    #Initialize Neural Networks
    print("Initialize NN")
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

    for ITER in range(MAXITER):
      print("Initiating Value Function evaluation for algo ", ALGO," Iter= ",ITER)
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
                    episodecost+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode)
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
      with open('objs.pkl', 'wb') as f:  # Python: open(..., 'w')
        pickle.dump(scenario, f)
      del scenario
      gc.collect()

      #Use Value Function table to incremmentally train Neural network
      # train the model using SGD
      print("[INFO] training network...")
      checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
      callbacks_list = [checkpoint]
      H = model.fit(Value_function_zeta, Value_function,epochs=epoch, batch_size=batch,   callbacks=callbacks_list,verbose=2)



      #Use NN parameters to formulate MIP for Policy Improvement
      global first_layer_weights
      global first_layer_biases
      global second_layer_weights
      global second_layer_biases
      global MHIGH
      global MLOW
      global L
      global U
      first_layer_weights = model.layers[0].get_weights()[0]
      first_layer_biases  = model.layers[0].get_weights()[1]
      second_layer_weights = model.layers[1].get_weights()[0]
      second_layer_biases  = model.layers[1].get_weights()[1]

      print("End of NN training")
    

      #Solve MIP and get new policy (z and epsilon)
     
      MHIGH = np.zeros(NUMBERNODES)
      MLOW = np.zeros(NUMBERNODES)
      LOW = 0
      UPPER = len(DESTC)+1
      L=np.zeros((NUMBERNODES,len(DESTC)))
      U=np.zeros((NUMBERNODES,len(DESTC)))

      for i in range(NUMBERNODES):
        for j in range(len(DESTC)):
          if first_layer_weights[j,i] < 0 :
            L[i,j]= UPPER
            U[i,j]= LOW
          else :
            L[i,j]= LOW
            U[i,j]= UPPER
   
      for i  in range(NUMBERNODES):
        MHIGH[i]= sum(first_layer_weights[j,i]*U[i,j] for j in range(len(DESTC))) + first_layer_biases[i]
        MLOW[i]= sum(first_layer_weights[j,i]*L[i,j] for j in range(len(DESTC))) + first_layer_biases[i]
      
      
      m = Model(name='BestOptz',log_output=True)
      #m.context.cplex_parameters.threads = 1
      m.context.cplex_parameters.timelimit = 2400
      #m.context.cplex_parameters.workmem= 256
      #m.context.cplex_parameters.mip.strategy.file=2
      #cplex.setParam(IloCplex::WorkMem, 1024);
      #cplex.setParam(IloCplex::NodeFileInd,2);
      print("Model set") 
     
      y = m.continuous_var_list(NUMBERNODES, name='y')
       
      s = m.binary_var_list(NUMBERNODES, name="s")
      
      z = m.continuous_var_list(len(DESTC), lb=1, ub=len(DESTC), name='z')
      
      x = m.binary_var_dict(((i, j) for i in range(len(DESTC)) for j in range(len(DESTC)) if i < j), name='x')
       
      obj = m.continuous_var(name='obj')
       
      m.set_objective("min", obj)
      #print("There goes variables")
       
      m.add_constraint_(obj >= sum(second_layer_weights[i,0]*y[i] for i in range(NUMBERNODES)) + second_layer_biases[0], ctname="const1" )
      #print("const1...")
       
      m.add_constraints_( (y[i] >= sum( first_layer_weights[j,i]*z[j] for j in range(len(DESTC)) )   + first_layer_biases[i] for i in range(NUMBERNODES) ))
      
      m.add_constraints_( (y[i] <= sum( first_layer_weights[j,i]*z[j] for j in range(len(DESTC)) )   + first_layer_biases[i]  - MLOW[i]*(1-s[i]) for i in range(NUMBERNODES)))
      
      m.add_constraints_( (y[i] <= MHIGH[i]*s[i] for i in range(NUMBERNODES)))

      #print("const2...")
      #No strengthen constraints

      m.add_constraints_( ( x[i,j]+x[j,k]+x[k,i]<=2 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j < k and k < i) )
      #print("const3...")
      
      #m.add_constraints_( ( -x[j,i]+x[j,k]+x[k,i]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j < k and k < i ))
      
      #m.add_constraints_( ( x[i,j]-x[k,j]+x[k,i]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j > k and k < i ))
      
      #m.add_constraints_( ( -x[j,i]-x[k,j]+x[k,i]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if  i > j and j > k and k < i))
      
      #m.add_constraints_( (x[i,j]+x[j,k]-x[i,k]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j < k and k > i ))
      
      #m.add_constraints_( ( -x[j,i]+x[j,k]-x[i,k]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j < k and k > i))
      
      #m.add_constraints_( ( x[i,j]-x[k,j]-x[i,k]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j > k and k > i))
      
      #m.add_constraints_( (-x[j,i]-x[k,j]-x[i,k]<=-1  for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j > k and k > i ))
      
      m.add_constraints_( ( z[i] == 1+sum(1-x[i,j] for j in range(len(DESTC)) if i < j ) + sum(x[j,i] for j in range(len(DESTC)) if i > j ) for i in range(len(DESTC)) ))
      
      m.register_callback(DOHeurCallback)
      cb = m.register_callback(DOLazyCallback)
      #cb.cts = []
      #cb.cts += [ -x[j,i]+x[j,k]+x[k,i]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j < k and k < i ]
      
      
      #cb.cts += [x[i,j]-x[k,j]+x[k,i]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j > k and k < i ]
      
      #cb.cts += [-x[j,i]-x[k,j]+x[k,i]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if  i > j and j > k and k < i]
      
      #cb.cts += [x[i,j]+x[j,k]-x[i,k]<=1 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j < k and k > i ]
      
      #cb.cts += [-x[j,i]+x[j,k]-x[i,k]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j < k and k > i]
      
      #cb.cts += [x[i,j]-x[k,j]-x[i,k]<=0 for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i < j and j > k and k > i]
      
      #cb.cts += [-x[j,i]-x[k,j]-x[i,k]<=-1  for i in range(len(DESTC)) for j in range(len(DESTC)) for k in range(len(DESTC)) if i > j and j > k and k > i ]
      #print("const4...") 
      #warmstart to guarantee at least one solution
      warmstart=m.new_solution()
      order_warmstart = sorted(DESTC)
      for i in range(len(DESTC)-1):
        for j in range(i+1,len(DESTC)):
         if DESTC.index(order_warmstart[i]) < DESTC.index(order_warmstart[j]):   warmstart.add_var_value(x[DESTC.index(order_warmstart[i]),DESTC.index(order_warmstart[j])], 1)
         else :
           warmstart.add_var_value(x[DESTC.index(order_warmstart[j]),DESTC.index(order_warmstart[i])], 0)
      m.add_mip_start(warmstart)
      
      #print("const5...")     
      msol=m.solve()
      #print("const6...")
      if msol.is_empty():
        print("The optbestz model was not solved correctly.")
        bestZ= [i for i in range(1,len(DESTC)+1)]       
        random.shuffle(bestZ)
        objZ= 0
        termination_status="NoSolution"
      else :
        bestZ = msol.get_values([z[ind] for ind in range(len(DESTC)) ])
        bestZ = [int(round(x)) for x in bestZ]
        objZ= msol.get_objective_value()
        termination_status="OK"
      print(bestZ," ,",objZ," Iter= ",ITER)
      # Getting back the objects:
      #global scenario
      with open('objs.pkl', 'rb') as f:  # Python : open(...)
        scenario = pickle.load(f)
    return bestZ, objZ


#run DRL 
start_time = time()
SOL_FSTAGE_REW1,SOLVALUE_REW1 =run_DRL(DESTC,CAPV,REWV,'DRL_REW1',NUMBERNODES,scenario,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
TIME_DRL_REW1= time()-start_time

start_time = time()
#SOL_FSTAGE_REW2H,SOLVALUE_REW2H =run_DRL(DESTC,CAPV,REWV,'DRL_REW2H',NUMBERNODES,scenario,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
TIME_DRL_REW2H= time()-start_time
SOL_FSTAGE_REW2H=SOL_FSTAGE_REW1

print("Will load Julia and Julia code now....wait")
import julia
jl = julia.Julia(compiled_modules=False)
from julia import Main
x = Main.include("juliacode.jl")
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
Main.NUMBERNODES = NUMBERNODES
print("Julia code loaded")

start_time = time()
SOL_FSTAGE_REW2E=SOL_FSTAGE_REW2H
#SOL_FSTAGE_REW2E,SOLVALUE_REW2E =run_DRL(DESTC,CAPV,REWV,'DRL_REW2E',NUMBERNODES,scenario,epsilon,learning_rate,MAXITER,LAST_SCENARIO_TO_TRAIN)
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
    RESULT_SIMU_REW1+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode)
    episode+=1
  

print("Simulating Value Funtion  for REWARD VARIANT 1", RESULT_SIMU_REW1)
input()

print("Simulating Value Funtion with random solution with REWARD VARIANT 1")

zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
#print(zetatest)
zetatest = [int(round(x)) for x in zetatest]
firststage = [0 for _ in range(len(DESTC))]
for i in range(len(DESTC)):
  firststage[zetatest[i]-1]= DESTC[i]
episode =FIRST_SCENARIO_TO_SIMULATE
RESULT_SIMU_REW1_RAND = 0
while episode < LAST_SCENARIO_TO_SIMULATE: 
  #print(episode," ")
  for _ in range(4):
    if episode%100==0:
      print(episode," ,")
    #calculate reward 1 for this scenario
    RESULT_SIMU_REW1_RAND+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode)
    episode +=1
  
print("Simulating Value Funtion  with random solution REWARD VARIANT 1", RESULT_SIMU_REW1_RAND)
##input()


##print("Simulating Value Funtion with Second random solution with REWARD VARIANT 1")

#zetatest = np.random.permutation(list(range(1,len(DESTC)+1)))
#print(zetatest)
#zetatest = [int(round(x)) for x in zetatest]
#firststage = [0 for _ in range(len(DESTC))]
#for i in range(len(DESTC)):
#  firststage[zetatest[i]-1]= DESTC[i]
#episode =FIRST_SCENARIO_TO_SIMULATE
#episodetotalreturn = 0
#while episode < LAST_SCENARIO_TO_SIMULATE: 
#  #print(episode," ")
#  for _ in range(4):
#    if episode%100==0:
#      print(episode," ,")
#    #calculate reward 1 for this scenario
#    episodetotalreturn+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode)
#    episode +=1
#print("Simulating Value Funtion  with Second random solution REWARD VARIANT 1", episodetotalreturn)
##input()


#print("Simulating Value Funtion with DRO")
#bestZ = SOL_FSTAGE_DRO
#print(bestZ)

#bestZ = [int(round(x)) for x in bestZ]

#firststage = [0 for _ in range(len(DESTC))]
#for i in range(len(DESTC)):
#  firststage[bestZ[i]-1]= DESTC[i]

#episode =FIRST_SCENARIO_TO_SIMULATE
#RESULT_SIMU_DRO = 0
#while episode < LAST_SCENARIO_TO_SIMULATE: 
#  #print(episode," ")
#  for _ in range(4):
#    if episode%100==0:
#      print(episode," ,")
#    #calculate reward 1 for this scenario
#    RESULT_SIMU_DRO+=calculate_reward1(A,DURV,REWV,CAPV,DESTC,firststage,episode)
#    episode+=1
  

#print("Simulating Value Funtion  with DRO", RESULT_SIMU_DRO)
##input()


#print("Simulation using result from best Value Function with REWARD VARIANT 2 EXACT")
#bestZ = SOL_FSTAGE_REW2E
#print(bestZ)
#firststage = [0 for _ in range(len(DESTC))]
#for i in range(len(DESTC)):
#  firststage[bestZ[i]-1]= DESTC[i]
#episode =FIRST_SCENARIO_TO_SIMULATE
#RESULT_SIMU_REW2E = 0
#TOTAL_OD_DECISIONS = 0
#TOTAL_OD_ALLOCATIONS = 0
#while episode < LAST_SCENARIO_TO_SIMULATE: 
#  #print(episode," ")
#    for _ in range(4):
#      if episode%100==0:
#        print(episode," ,")
#      #calculate reward 2 exact for this scenario 
#      Main.scenario = scenario[episode]
#      Main.firststage = firststage
#      RES1,RES3,RES2= jl.eval("rewardvariant2(V,A,REWV,PRICEOD,DESTC,scenario,firststage)")
#      RESULT_SIMU_REW2E+= RES2
#      TOTAL_OD_DECISIONS += sum(scenario[episode][len(DESTC):2*len(DESTC)])
#      TOTAL_OD_ALLOCATIONS += RES1
#      episode +=1
      
#print("Simulation using result from best Value Function with REWARD VARIANT 2 EXACT ", RESULT_SIMU_REW2E," ,",TOTAL_OD_ALLOCATIONS," ,",TOTAL_OD_DECISIONS)
##input()

#print("Simulating Value Funtion with REWARD VARIANT 2 HEUR")
#bestZ = SOL_FSTAGE_REW2H
#print(bestZ)
#firststage = [0 for _ in range(len(DESTC))]
#for i in range(len(DESTC)):
#  firststage[bestZ[i]-1]= DESTC[i]

#episode =FIRST_SCENARIO_TO_SIMULATE
#RESULT_SIMU_REW2H = 0
#while episode < LAST_SCENARIO_TO_SIMULATE: 
#  #print(episode," ")
#  for _ in range(4):
#    if episode%10==0:
#      print(episode," ,")
#    #calculate reward 2 heur for this scenario 
#    RESULT_SIMU_REW2H+=calculate_reward2heuristic(A,DURV,REWV,CAPV,DESTC,firststage,scenario,episode)
#    episode +=1
#print("Simulation using result from best Value Function with REWARD VARIANT 2 HEUR ", RESULT_SIMU_REW2H)
##input()



#print("Simulating Reoptimization: considers Variant 1 type reward")
#episode =FIRST_SCENARIO_TO_SIMULATE
#RESULT_SIMU_REOPT = 0
#while episode < LAST_SCENARIO_TO_SIMULATE: 
#  #print(episode," ")
#  episodecost = 0
#  for _ in range(4):
#    if episode%100==0:
#      print(episode," ,")
#    Main.scenario = scenario[episode]
#    RESULT_SIMU_REOPT += jl.eval("reoptimization(V,A,REWV,PRICEOD,DESTC,scenario)")
#    episode +=1
#  RESULT_SIMU_REOPT+=episodecost
#  print(episode," : ",RESULT_SIMU_REOPT)
#print("Simulation using result from Reoptimization ", RESULT_SIMU_REOPT)
##input()

data = [' ',inst_id,ALGO,MAXITER,epsilon,learning_rate,NUMBERNODES,epoch,batch,LAST_SCENARIO_TO_TRAIN,
SOLVALUE_REW1,TIME_DRL_REW1,SOLVALUE_REW2E,TIME_DRL_REW2E, SOLVALUE_REW2H,TIME_DRL_REW2H,SOLVALUE_DRO,TIME_DRO,RESULT_SIMU_REW1,RESULT_SIMU_REW1_RAND,RESULT_SIMU_REW2E,RESULT_SIMU_REW2H,RESULT_SIMU_DRO,RESULT_SIMU_REOPT, TOTAL_OD_DECISIONS,TOTAL_OD_ALLOCATIONS]

writer.writerow(data)
f.close()

