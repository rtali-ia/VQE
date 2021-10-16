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

class Entry(base):
    __table_args__ = {'schema' : 'Logs', 'extend_existing': True}
    __tablename__ = 'iterlog'
    logid = Column(Integer, primary_key=True)
    owner = Column(String)
    n_qubits = Column(Integer)
    g = Column(Numeric)
    layers = Column(Integer)
    ETA = Column(Numeric)
    MAX_ITER = Column(Integer)
    init_type = Column(String)
    NUM_ROUNDS = Column(Integer)
    iter = Column(Integer)
    overlap = Column(Numeric)
    energy = Column(Numeric)
    norm_grad = Column(Numeric)
    vector = Column(String)
    angles = Column(String)
    log_start_time = Column(TIMESTAMP)
    atype = Column(String)

class Round(base):
    __table_args__ = {'schema' : 'Logs', 'extend_existing': True}
    __tablename__ = 'roundlog'
    logid = Column(Integer, primary_key=True)
    owner = Column(String)
    n_qubits = Column(Integer)
    g = Column(Numeric)
    layers = Column(Integer)
    ETA = Column(Numeric)
    MAX_ITER = Column(Integer)
    init_type = Column(String)
    NUM_ROUNDS = Column(Integer)
    round = Column(Integer)
    overlap = Column(Numeric)
    energy = Column(Numeric)
    ansatz = Column(String)
    params = Column(String)
    log_start_time = Column(TIMESTAMP)
    round_time = Column(Numeric)
    atype = Column(String)
    
    
    
##################################
      #Define Basics
##################################

# single qubit basis states |0> and |1>
q0 = np.array([[1],[0]])
q1 = np.array([[0],[1]])

# Pauli Matrices
I  = np.array([[ 1, 0],[ 0, 1]])
X = np.array([[ 0, 1],[ 1, 0]])
Y = np.array([[ 0,-1j],[1j, 0]])
Z = np.array([[ 1, 0],[ 0,-1]])
HG = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
CZ = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])



#Creates the all zero input state.                            
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
    
    
                           

#Creates the equally superimposed product state from all zero state.                           
def equal_Superposition(n_qubits, init_all_zero):
        
    if n_qubits < 2:
        return 'Invalid Input : Specify at least 2 qubits'
            
    else:
            
        all_H = np.kron(HG,HG)
            
        for t in range(n_qubits - 2):
            all_H = np.kron(all_H,HG)
            
        equal_Superpos = all_H@init_all_zero 
                
        return equal_Superpos
    
#Analytical Ground State using Numpy's inbuilt eigh function.
def get_analytical_ground_state(H):
  e, v = LA.eigh(H)
  return np.min(e), v[:,np.argmin(e)]


#Create Unitary
def CU(Q, theta, n_qubits):
  Id = np.eye(2**n_qubits)
  return np.cos(theta)*Id - 1j*np.sin(theta)*Q



#Define Real Ansatz
def ansatz_real(YZ_param_set, Y_param_set , real_comps, n_qubits, layers):
    
  #Initialize Ansatz to I
  ansatz = np.eye(2**n_qubits)
  
  YZ_components = real_comps[0]
  Y_components = real_comps[1]

  for layer in range(layers):
    
    for ct1, comp1 in enumerate(YZ_components):
      ansatz = CU(comp1, theta = YZ_param_set[layer],n_qubits=n_qubits)@ansatz

    for ct2, comp2 in enumerate(Y_components):
      ansatz = CU(comp2,  theta = Y_param_set[layer],n_qubits=n_qubits)@ansatz

  return ansatz




##################################
      #Define TFIM model
##################################

#helper funnction for TFIM model creation.
def component_sums(components, real_comps, n_qubits):

  ZZ_sum = np.zeros((2**n_qubits,2**n_qubits))
  X_sum = np.zeros((2**n_qubits,2**n_qubits))
  Y_sum = np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
  YZ_sum = np.zeros((2**n_qubits,2**n_qubits),dtype=complex)
  
  for zz_arr in components[0]:
    ZZ_sum += zz_arr

  for x_arr in components[1]:
    X_sum += x_arr
    
  for yz_arr in real_comps[0]:
    YZ_sum += yz_arr

  for y_arr in real_comps[1]:
    Y_sum += y_arr

  return ZZ_sum, X_sum, Y_sum, YZ_sum



def array_coding_to_kron(arr, type):
  n_qubits = len(arr)
  
  if type == 'ZZ':
    convert = {0 : I, 1 : Z} #Dictionary that maps code to Pauli Matrix
    expr = np.kron(convert[arr[0]],convert[arr[1]])
    for t in range(2, n_qubits):
      expr = np.kron(expr,convert[arr[t]])

    return expr

  elif type == 'YZ':
    convert = {0 : I, 1 : Z, -1 : Y} #Dictionary that maps code to Pauli Matrix
    expr4 = np.kron(convert[arr[0]],convert[arr[1]])
    for p in range(2, n_qubits):
      expr4 = np.kron(expr4,convert[arr[p]])

    return expr4

  elif type == 'Y':
    convert = {0 : I, 1 : Y}
    expr3 = np.kron(convert[arr[0]],convert[arr[1]])
    for s in range(2, n_qubits):
      expr3 = np.kron(expr3,convert[arr[s]])
      
    return expr3

  else:
    convert = {0 : I, 1 : X}
    expr2 = np.kron(convert[arr[0]],convert[arr[1]])
    for k in range(2, n_qubits):
      expr2 = np.kron(expr2,convert[arr[k]])

    return expr2



def create_TFIM(n_qubits, g):

  if n_qubits == 2:
    return -1*np.kron(Z,Z) -g*(np.kron(X,I)+np.kron(I,X)), {0: [np.kron(Z,Z)], 1: [np.kron(X,I),np.kron(I,X)]}, {0: [np.kron(Y,Z)], 1: [np.kron(Y,I), np.kron(I,Y)]}

  else:
    #This will store all the kronecker products used in Ansatz Layers
    comps = {0:[], 1:[]}
    real_comps = {0:[], 1:[]}

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
      
      
    #Y Terms
    for i in range(n_qubits):
        y_arr = np.zeros(n_qubits)
        y_arr[i]  = 1
        
        #Call encoding and append
        real_comps[1].append(array_coding_to_kron(y_arr,type='Y'))
        
    #YZ terms
    for i in range(n_qubits):
      yz_arr = np.zeros(n_qubits)
      if i < n_qubits - 1:
        yz_arr[i] = -1          #-1 means Y
        yz_arr[i+1] = 1         # 1 means Z
      else:
        yz_arr[0] = 1
        yz_arr[i] = -1
        
    #Append component
      real_comps[0].append(array_coding_to_kron(yz_arr,type='YZ'))

    return tfim, comps, real_comps



#This funaction calculates the overlap of the solution with analytical ground state.
def overlap_calculator(min_pm, ground_st):
    return np.abs(np.vdot(min_pm, ground_st))**2
    
def power_computation(H, circuit_input):
    return (1/(LA.norm(H@circuit_input)))*(H@circuit_input)

def energy_raw(H,psi):
    return np.real((psi.conj().T)@H@psi)[0][0]

def get_eta(eta_in, grad_prev, grad_now):
    return eta_in*grad_now/grad_prev




############

#Gradient

############

def all_Y(Y_components,param,n_qubits):
  Y = np.eye(2**n_qubits)
  for component in Y_components:
    Y = CU(component,param,n_qubits=n_qubits)@Y
  return Y


def all_YZ(YZ_components,param,n_qubits):
  YZ = np.eye(2**n_qubits)
  for component in YZ_components:
    YZ = CU(component,param,n_qubits=n_qubits)@YZ
  return YZ


#Gradient - Harrow Napp
def grad_power(b0,b, YZ_param_set, Y_param_set, components, real_comps, n_qubits,layers):

  #Prepare the common right hand side for the derivative
  psi_right = b0@(b.conj().T)
  
  #Prepare fixed parts of the overall derivative
  anz = ansatz_real(YZ_param_set, Y_param_set, components, n_qubits, layers)
  rpart = np.real((b.conj().T)@anz@b0)
  ipart = np.imag((b.conj().T)@anz@b0)
  

  #Sum the ZZ and X components
  sum_ZZ, sum_X, sum_Y, sum_YZ = component_sums(components, real_comps, n_qubits=n_qubits) #This is implemented via a function call.
  
  #Total parameters
  param_per_layer =  2 # We always have 2 params per layer for VHA Ansatz.
  full_derivative = np.zeros(2*layers) # This is just initialization for the gradient vector

  #Derivative Expression for each param

  #ZZ params

  #Loop through all YZ params
  for j in range(layers):
    #initialize computation for the jth ZZ derivative
    deriv = np.eye(2**n_qubits)

    #This inner loop is to loop through the circuit elements, only one of the YZ elements will have a derivative 
    for i in range(layers):

      all_Ys = all_Y(components[1], Y_param_set[i], n_qubits=n_qubits)
      all_YZs = all_YZ(components[0], YZ_param_set[i], n_qubits=n_qubits)

      if i == j:
        deriv = all_Ys@all_YZs@(1j*sum_YZ)@deriv
      else:
        deriv = all_Ys@all_YZs@deriv
        
    trace_deriv = np.trace(deriv@psi_right)

    #Store
    full_derivative[j*param_per_layer] = (2*rpart*np.real(trace_deriv)+2*ipart*np.imag(trace_deriv))[0][0]


  #X params
  for k in range(layers):
    #initialize computation for the kth X derivative
    deriv2 = np.eye(2**n_qubits)

    #This inner loop is to loop through the circuit elements, only one of the X elements will have a derivative 
    for l in range(layers):
    
      all_YZs = all_YZ(components[0], YZ_param_set[l], n_qubits=n_qubits)
      all_Ys = all_Y(components[1],Y_param_set[l], n_qubits=n_qubits)
  
      if l == k:
        deriv2 = all_Ys@(1j*sum_Y)@all_YZs@deriv2
      else:
        deriv2 = all_Ys@all_YZs@deriv2
        
    trace_deriv2 = np.trace(deriv2@psi_right)

    #Store
    full_derivative[k*param_per_layer+1] = (2*rpart*np.real(trace_deriv2)+2*ipart*np.imag(trace_deriv2))[0][0]


  #Return all partial derivatives
  return full_derivative


def grad_positioning(grad):
    YZ = []
    Y = []
    for i in range(len(grad)):
        if i%2 == 0:
            YZ.append(grad[i])
        else:
            Y.append(grad[i])
    return np.array(YZ), np.array(Y)

def grad_descent(b0, b, components, real_comps, YZ_param_set, Y_param_set, MAXITERS, eta, GRADTOL, n_qubits, layers, time_start, log_freq = 1, plotting = 'off', logging = 'off'):
    
    store_grad_norm = []

    #Theta is a vector ---> np.array
    theta_Y = Y_param_set.copy() 
    theta_YZ = YZ_param_set.copy()

    #Keep track of number of iterations
    counter = 0 

    #Iterate
    for iter in range(MAXITERS):

        grad = grad_power(b0,b,YZ_param_set=theta_YZ, Y_param_set=theta_Y, components=components, real_comps = real_comps,
                            n_qubits=n_qubits, layers=layers)
    
        if LA.norm(grad) < GRADTOL:
            break

        #Extract components - This is to correctly order gradient components
        YZ, Y = grad_positioning(grad)

        #Update thetas - Grad Ascent
        theta_YZ = theta_YZ - eta*YZ
        theta_Y = theta_Y - eta*Y
    
        #Eigenvector
        v = ansatz_real(YZ_param_set = theta_YZ, Y_param_set = theta_Y, real_comps=real_comps, n_qubits=n_qubits, 
                   layers=layers)@b0
        
        #Overlap
        ov = overlap_calculator(v,v_an)
        
        #Energy
        ev = energy_raw(H,v)

        
    
    #Some Periodic Logging on Terminal for large N --> if requested.
        if logging == 'on':
            #Log every 20 steps.
            if counter%log_freq == 0:
                vals_now = Entry(owner = 'R', n_qubits = n_qubits, g = g, layers = layers, ETA = ETA, MAX_ITER = MAX_ITER, NUM_ROUNDS = ROUNDS,
                                init_type = btype, iter = counter, overlap = ov, energy = ev, 
                                  norm_grad = LA.norm(grad) , vector = str(v), angles = str([theta_YZ,theta_Y]), 
                                    log_start_time = time_start, 
                                        atype = 'real')  
    
                session.add(vals_now)  
                session.commit()
                
        #Keep track of number of iterations
        counter += 1

    #Store Gradient Norm and Energy
        store_grad_norm.append(LA.norm(grad))
  
  #Some Plotting --> if requested.
    if plotting == 'on':
        plt.plot(range(counter),store_grad_norm)
        plt.title('Track Gradient Norm')
        plt.xlabel('Iteratio Number')
        plt.ylabel('L2 Norm of the Gradient')
        plt.show()

    return [theta_YZ,theta_Y], counter, v, LA.norm(grad), store_grad_norm





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


OFFSET_DICT = {4:6, 6:10, 8:20}
LAYERS_DICT = {4: [2,4],6:[3,5,6],8:[4,6,8,10]}
OWNER = 'R'                                                       # For Justin = 'J', For Ronak = 'R'
tstart = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')         
print(tstart)                                                     # Important - We identify a run by this value. Save this somewhere.

for n_qubits in [4,6,8]:
    print('Executing now @ Number of Qubits = ',n_qubits)                         # This is to track progress of where the execution is - N QUBITS
    for g in [0.5,1.0,2.0]:
      print('Executing Now @ g = ',g)                                             # This is to track progress of where the execution is - G
      for btype in ['eq','rnd']:
        for ETA in [0.001, 0.01, 0.1]:
          for MAX_ITER in [10,20,50]:
            for ROUNDS in [5,10]:
              for layers in LAYERS_DICT[n_qubits]:                               
                ov = 0                                            # Just an initialization. Don't change
                MAXIMUM = 0                                       # Just an initialization. Don't change
                TOL = 0.0001                                      # Just an initialization. Don't change
                
                #Create the TFIM Model
                H, components, real_comps = create_TFIM(n_qubits = n_qubits, g = g)
                
                YZ_param_set = (pi/3)*np.ones(p) 
                Y_param_set = (pi/3)*np.ones(p)
                
                
                #get analytical ground state
                e_an, v_an = get_analytical_ground_state(H)     #Find the actual algebraic ground state.
                
                
                b0_now = b0_e #or b0 or b0_r                    #Select psi0 for our ansatz. One of the three choices.
                
                if btype == 'eq':
                  b0_e = equal_Superposition(n_qubits,all_Zero_State(n_qubits))
                else:
                  b0_e = psi0(n_qubits)
                  
                b0_now = b0_e
                
                
                Hprime = H - OFFSET_DICT[n_qubits]*np.eye(2**n_qubits)
                in_vec = Hprime@b0_now/LA.norm(H@b0_now)

                while(ov <= 0.99999 and MAXIMUM <= ROUNDS):
                    round_start_time = time.time()
                    p, _, vec, grad, g_hist = grad_descent(b0_e, in_vec, components, real_comps, YZ_param_set, Y_param_set, MAX_ITER, ETA, TOL, n_qubits, layers, time_start=tstart ,plotting = 'off', logging = 'on')
                    round_end_time = time.time()
                    round_time = (round_end_time - round_start_time)/60
                    ov = overlap_calculator(vec,v_an)
                    ev = energy_raw(H,vec)
                    in_vec = Hprime@vec/LA.norm(Hprime@vec)
    
    
                    #Log at round level
                    vals_round = Round(owner = OWNER, n_qubits = n_qubits, g = g, layers = layers, ETA = ETA, MAX_ITER = MAX_ITER, NUM_ROUNDS = ROUNDS,
                                      init_type = btype, round = MAXIMUM, overlap = ov, energy = ev, 
                                      ansatz = str(vec), params = str(p), log_start_time = tstart, round_time = round_time , atype = 'real')  
    
                    session.add(vals_round)  
                    session.commit()

                    MAXIMUM += 1
    
    
#Close Database Session
session.close()