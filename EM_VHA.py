#%%
############################################
#Iterative Power Method for Real Ansatz
############################################

############################################
#Author - R Tali [rtali@iastate.edu]
#Version - v2.2
############################################

import numpy as np
import pandas as pd
import logging
import random
from numpy import pi
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time
from datetime import datetime
plt.style.use('fivethirtyeight')


import datetime as dt
from sqlalchemy import create_engine  
from sqlalchemy import Table, Column, String, MetaData
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql.sqltypes import TIMESTAMP, Integer, Numeric

###################################################
# Define ORM Adaptor
###################################################

class EMEntry(base):
    __table_args__ = {'schema' : 'Logs', 'extend_existing': True}
    __tablename__ = 'emlog'
    logid = Column(Integer, primary_key=True)
    owner = Column(String)
    n_qubits = Column(Integer)
    g = Column(Numeric)
    layers = Column(Integer)
    ETA = Column(Numeric)
    MAX_ITER = Column(Integer)
    init_type = Column(String)
    iter = Column(Integer)
    overlap = Column(Numeric)
    energy = Column(Numeric)
    norm_grad = Column(Numeric)
    vector = Column(String)
    angles = Column(String)
    log_start_time = Column(TIMESTAMP)
    atype = Column(String)

class EMSummary(base):
    __table_args__ = {'schema' : 'Logs', 'extend_existing': True}
    __tablename__ = 'emsummary'
    logid = Column(Integer, primary_key=True)
    owner = Column(String)
    n_qubits = Column(Integer)
    g = Column(Numeric)
    layers = Column(Integer)
    ETA = Column(Numeric)
    MAX_ITER = Column(Integer)
    init_type = Column(String)
    NUMITERS = Column(Integer)
    overlap = Column(Numeric)
    energy = Column(Numeric)
    vector = Column(String)
    angles = Column(String)
    log_start_time = Column(TIMESTAMP)
    total_time = Column(Numeric)
    atype = Column(String)



# single qubit basis states |0> and |1>
q0 = np.array([[1],[0]])
q1 = np.array([[0],[1]])

#Init State
init_all_zero = np.kron(np.kron(np.kron(q0,q0),q0),q0)

# Pauli Matrices
I  = np.array([[ 1, 0],[ 0, 1]])
X = np.array([[ 0, 1],[ 1, 0]])
Y = np.array([[ 0,-1j],[1j, 0]])
Z = np.array([[ 1, 0],[ 0,-1]])
HG = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
                             

def all_Zero_State(n_qubits):
        
   if n_qubits < 2:
       return 'Invalid Input : Specify at least 2 qubits'
            
   else:
            #Init State
       init_all_zero = np.kron(q0,q0)
            
       for t in range(n_qubits - 2):
           init_all_zero = np.kron(init_all_zero,q0)
                
       return init_all_zero                            
                             


#Creates Random Initial State Ansatz
def psi0(n_qubits):
    
    if n_qubits < 2:
       return 'Invalid Input : Specify at least 2 qubits'
            
    else:
       pick = random.uniform(0, 2*pi)
       i1 = np.cos(pick)*q0 + np.sin(pick)*q1
       pick = random.uniform(0, 2*pi)
       i2 = np.cos(pick)*q0 + np.sin(pick)*q1
       init_random = np.kron(i1,i2)
            
       for t in range(n_qubits - 2):
           pick = random.uniform(0, 2*pi)
           inow = np.cos(pick)*q0 + np.sin(pick)*q1
           init_random = np.kron(init_random, inow)
                
       return init_random

def equal_Superposition(n_qubits, init_all_zero):
        
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'
            
    else:
            
        all_H = np.kron(HG,HG)
            
        for t in range(n_qubits - 2):
            all_H = np.kron(all_H,HG)
            
        equal_Superpos = all_H@init_all_zero 
                
        return equal_Superpos
    
    
#Analytical Ground State
def get_analytical_ground_state(H):
  e, v = LA.eigh(H)
  return np.min(e), v[:,np.argmin(e)]


#Create Unitary
def CU(Q, theta, n_qubits):
  Id = np.eye(2**n_qubits)
  return np.cos(theta)*Id - 1j*np.sin(theta)*Q

#Ansatz Circuit 


def ansatz_vha(X_param_set, ZZ_param_set, components, n_qubits, layers):

  #Initialize Ansatz to I
  ansatz = np.eye(2**n_qubits)
  
  ZZ_components = components[0]
  X_components = components[1]

  for layer in range(layers):
    
    for ct1, comp1 in enumerate(ZZ_components):
      ansatz = CU(comp1, theta = ZZ_param_set[layer],n_qubits=n_qubits)@ansatz

    for ct2, comp2 in enumerate(X_components):
      ansatz = CU(comp2,  theta = X_param_set[layer],n_qubits=n_qubits)@ansatz

  return ansatz



#This funaction calculates the overlap of the solution with analytical ground state.
def overlap_calculator(min_pm, ground_st):
    return np.abs(np.vdot(min_pm, ground_st))**2
    
def power_computation(H, circuit_input):
    return (1/(LA.norm(H@circuit_input)))*(H@circuit_input)

def energy_raw(H,psi):
    return np.real((psi.conj().T)@H@psi)[0][0]

def get_eta(eta_in, grad_prev, grad_now):
    return eta_in*grad_now/grad_prev

#Expectation
def energy_VHA(H,components, circuit_input, X_param_set, ZZ_param_set, n_qubits , layers):
  psi = ansatz_vha(X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, components = components, 
                   n_qubits = n_qubits, layers=layers)@circuit_input
  return np.real((psi.conj().T)@H@psi)[0][0]



#Define TFIM model

def component_sums(components, n_qubits):

  ZZ_sum = np.zeros((2**n_qubits,2**n_qubits))
  X_sum = np.zeros((2**n_qubits,2**n_qubits))

  for zz_arr in components[0]:
    ZZ_sum += zz_arr

  for x_arr in components[1]:
    X_sum += x_arr

  return ZZ_sum, X_sum

def array_coding_to_kron(arr, type):
  n_qubits = len(arr)
  
  if type == 'ZZ':
    convert = {0 : I, 1 : Z} #Dictionary that maps code to Pauli Matrix
    expr = np.kron(convert[arr[0]],convert[arr[1]])
    for t in range(2, n_qubits):
      expr = np.kron(expr,convert[arr[t]])

    return expr

  else:
    convert = {0 : I, 1 : X}
    expr2 = np.kron(convert[arr[0]],convert[arr[1]])
    for k in range(2, n_qubits):
      expr2 = np.kron(expr2,convert[arr[k]])

    return expr2

def create_TFIM(n_qubits, g):

  if n_qubits == 2:
    return -1*np.kron(Z,Z) -g*(np.kron(X,I)+np.kron(I,X)), {0: [np.kron(Z,Z)], 1: [np.kron(X,I),np.kron(I,X)]}

  else:
    #This will store all the kronecker products used in Ansatz Layers
    comps = {0:[],1:[]}

    #Initializing an empty 
    tfim = np.zeros((2**n_qubits,2**n_qubits))

    # Encode ZZ Terms
    for i in range(n_qubits):
      zz_arr = np.zeros(n_qubits)
      if i < n_qubits - 1:
        zz_arr[i] = 1
        zz_arr[i+1] = 1
      else:
        zz_arr[0] = 1
        zz_arr[i] = 1

      #Call the coding function
      tfim = tfim - array_coding_to_kron(zz_arr,type='ZZ')
      #Append component
      comps[0].append(array_coding_to_kron(zz_arr,type='ZZ'))

    #X Terms
    for i in range(n_qubits):
      x_arr = np.zeros(n_qubits)
      x_arr[i] = 1

      #Call the coding function
      tfim = tfim -g* array_coding_to_kron(x_arr,type='X')
      #Append component
      comps[1].append(array_coding_to_kron(x_arr,type='X'))

    return tfim, comps



#Harrow Napp

#Helper functions to compute derivative

def all_X(X_components,param,n_qubits):
  X = np.eye(2**n_qubits)
  for component in X_components:
    X = CU(component,param,n_qubits=n_qubits)@X
  return X


def all_ZZ(ZZ_components,param,n_qubits):
  ZZ = np.eye(2**n_qubits)
  for component in ZZ_components:
    ZZ = CU(component,param,n_qubits=n_qubits)@ZZ
  return ZZ


#Gradient - Harrow Napp
def grad_harrow_napp(H, X_param_set, ZZ_param_set,components, circuit_input, n_qubits,layers):

  #Prepare the common right hand side for the Harrow Napp Expression
  H_psi_right = H@ansatz_vha(X_param_set = X_param_set, ZZ_param_set = ZZ_param_set, 
                             components = components, n_qubits = n_qubits, 
                             layers = layers)@circuit_input


  #Sum the ZZ and X components
  sum_ZZ, sum_X = component_sums(components, n_qubits=n_qubits) #This is implemented via a function call.
  
  #Total parameters
  param_per_layer =  2
  full_derivative = np.zeros(2*layers) #This is just initialization for the gradient vector

  #Derivative Expression for each param

  #ZZ params

  #Loop through all ZZ params
  for j in range(layers):
    #initialize computation for the jth ZZ derivative
    psi_left_d_ZZ = circuit_input

    #This inner loop is to loop through the circuit elements, only one of the ZZ elements will have a derivative 
    for i in range(layers):

      all_Xs = all_X(components[1],X_param_set[i],n_qubits=n_qubits)
      all_ZZs = all_ZZ(components[0],ZZ_param_set[i],n_qubits=n_qubits)

      if i == j:
        psi_left_d_ZZ = all_Xs@all_ZZs@sum_ZZ@psi_left_d_ZZ
      else:
        psi_left_d_ZZ = all_Xs@all_ZZs@psi_left_d_ZZ

    #Store
    full_derivative[j*param_per_layer] = -2*np.imag((psi_left_d_ZZ.conj().T)@H_psi_right)


  #X params
  for k in range(layers):
    #initialize computation for the kth X derivative
    psi_left_d_X = circuit_input

    #This inner loop is to loop through the circuit elements, only one of the X elements will have a derivative 
    for l in range(layers):
    
      all_ZZs = all_ZZ(components[0],ZZ_param_set[l],n_qubits=n_qubits)
      all_Xs = all_X(components[1],X_param_set[l],n_qubits=n_qubits)
  
      if l == k:
        psi_left_d_X = all_Xs@sum_X@all_ZZs@psi_left_d_X
      else:
        psi_left_d_X = all_Xs@all_ZZs@psi_left_d_X

    #Store
    full_derivative[k*param_per_layer+1] = -2*np.imag((psi_left_d_X.conj().T)@H_psi_right)

  #Return all partial derivatives
  return full_derivative


def grad_positioning(grad):
  ZZ = []
  X = []
  for i in range(len(grad)):
    if i%2 == 0:
      ZZ.append(grad[i])
    else:
      X.append(grad[i])
  return np.array(ZZ), np.array(X)


def get_eta(eta_in, grad_prev, grad_now):
    return eta_in*grad_now/grad_prev
    

def hn_grad_desc_quantum(H, components, X_param_set, ZZ_param_set, circuit_input, MAXITERS, ETA, GRADTOL, n_qubits, layers, time_start, log_freq = 1, plotting = 'off', logging = 'on', adapt=False):

  store_grad_norm = []
  store_energy = []
  eta_now = ETA
  
  #Theta is a vector - np.array
  theta_X = X_param_set.copy() 
  theta_ZZ = ZZ_param_set.copy()
  
  #Keep track of number of iterations
  counter = 0 

  #Iterate
  for iter in range(MAXITERS):

    grad = grad_harrow_napp(H=H,X_param_set=theta_X,ZZ_param_set=theta_ZZ,components=components,
                            circuit_input=circuit_input ,n_qubits=n_qubits,layers=layers)
    
    if LA.norm(grad) < GRADTOL:
      break

    #Extract components - This is to correctly order gradient components
    ZZ, X = grad_positioning(grad)
    
    if counter > 0 and adapt==True:
        eta_now = get_eta(eta_now,store_grad_norm[-1],LA.norm(grad))
    else:
        eta_now = ETA

    #Update thetas
    theta_ZZ = theta_ZZ - eta_now*ZZ
    theta_X = theta_X - eta_now*X
    
    #Eigenvector
    v = ansatz_vha(X_param_set = theta_X, ZZ_param_set = theta_ZZ, components=components, n_qubits=n_qubits, layers=layers)@circuit_input

    #Eigenvalue
    e = energy_VHA(H = H ,components = components, circuit_input = circuit_input, X_param_set=theta_X, ZZ_param_set=theta_ZZ, 
                   n_qubits = n_qubits, layers = layers)

    
    
    if logging == 'on':
        if counter%log_freq == 0:
            vals_now = EMEntry(owner = 'R', n_qubits = n_qubits, g = g, layers = layers, ETA = ETA, MAX_ITER = MAX_ITER,
                                init_type = btype, iter = counter, overlap = ov, energy = ev, 
                                  norm_grad = LA.norm(grad) , vector = str(v), angles = str([theta_ZZ,theta_X]), 
                                    log_start_time = time_start, 
                                        atype = 'hva')  
    
            session.add(vals_now)  
            session.commit()
            
    #Keep track of number of iterations
    counter += 1
    
    #Store Gradient Norm and Energy
    store_grad_norm.append(LA.norm(grad))
    store_energy.append(e)
  
  
  #Some Plotting
  if plotting == 'on':
    plt.plot(range(counter),store_grad_norm)
    plt.title('Track Gradient Norm')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Norm of the Gradient')
    plt.show()

    plt.plot(range(counter),store_energy)
    plt.title('Track Cost Function')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Minimum Eigen Value attained')
    plt.show()

  return [theta_ZZ,theta_X], counter, e, v, LA.norm(grad)




#%%


##################################
# Logging in Database
##################################

try:

    db = create_engine('postgresql://postgres:root@localhost:5432/postgres')
    base = declarative_base()
    
    Session = sessionmaker(db)  
    session = Session()

except Exception as e:
    
    print(e)
    


LAYERS_DICT = {4: [2,4],6:[3,5,6],8:[4,6,8,10]}
OWNER = 'R'                                                       # For Justin = 'J', For Ronak = 'R'
tstart = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')         
print(tstart)                                                     # Important - We identify a run by this value. Save this somewhere.

for n_qubits in [4,6,8]:
    print('Executing now @ Number of Qubits = ',n_qubits)         # This is to track progress of where the execution is - N QUBITS
    for g in [0.5,1.0,2.0]:
      print('Executing Now @ g = ',g)                             # This is to track progress of where the execution is - G
      for btype in ['eq','rnd']:
        for ETA in [0.001, 0.01, 0.1]:
          for MAX_ITER in [10,20,50]:
              for layers in LAYERS_DICT[n_qubits]:                               
                TOL = 0.0001                                      # Just an initialization. Don't change
                
                #Create the TFIM Model
                H, components = create_TFIM(n_qubits = n_qubits, g = g)
                
                ZZ_param_set = (pi/3)*np.ones(p) 
                X_param_set = (pi/3)*np.ones(p)
                
                
                #get analytical ground state
                e_an, v_an = get_analytical_ground_state(H)     #Find the actual algebraic ground state.
                
                
                b0_now = b0_e #or b0 or b0_r                    #Select psi0 for our ansatz. One of the three choices.
                
                if btype == 'eq':
                  b0_e = equal_Superposition(n_qubits,all_Zero_State(n_qubits))
                else:
                  b0_e = psi0(n_qubits)
                  
    
             
                round_start_time = time.time()
                p, cnt , eig, vec, grad = hn_grad_desc_quantum(H, components, X_param_set, ZZ_param_set, b0_e, MAX_ITER, ETA, TOL, n_qubits, layers, time_start=tstart ,plotting = 'off', logging = 'on')
                round_end_time = time.time()
                round_time = (round_end_time - round_start_time)/60
                
                ov = overlap_calculator(vec,v_an)
                ev = energy_raw(H,vec)
                
    
    
                #Log at round level
                vals_round = EMSummary(owner = OWNER, n_qubits = n_qubits, g = g, layers = layers, ETA = ETA, MAX_ITER = MAX_ITER,
                                      init_type = btype, NUMITERS = cnt, overlap = ov, energy = ev, 
                                      ansatz = str(vec), params = str(p), log_start_time = tstart, total_time = round_time , atype = 'hva')  
    
                session.add(vals_round)  
                session.commit()

                



#Close Database Session
session.close()
# %%
