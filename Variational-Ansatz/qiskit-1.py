# -*- coding: utf-8 -*-
"""
Implementing VQA Ansatz
"""

#%%

TOKEN = 'f634b2a31452a1ee1d3693684e2a09e0bf30b4a581e90417a04c1cea9c356974562c7b45e945f864c9998376e56ee8c52f243c2360e91c2d4124157456ab4257'

#%%

import numpy as np
from numpy import pi
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.aqua.utils import *

#%%

# Loading your IBM Quantum account(s)
provider = IBMQ.enable_account(TOKEN=TOKEN)


#%%

#Define the Hamiltonian
I  = np.array([[ 1, 0],[ 0, 1]])
X = np.array([[ 0, 1],[ 1, 0]])
Y = np.array([[ 0,-1j],[1j, 0]])
Z = np.array([[ 1, 0],[ 0,-1]])

h = 0.5
H = -tensorproduct(Z,Z,I,I) - tensorproduct(I,Z,Z,I) - tensorproduct(I,I,Z,Z)
H = H - h*(tensorproduct(X,I,I,I) + tensorproduct(I,X,I,I) + tensorproduct(I,I,X,I) + tensorproduct(I,I,I,X))

#%%

#Direct Diagonalization

from numpy import linalg as LA
w, v = LA.eig(H)
print(w)
print(v)

#%%

def circuit_var(beta, gamma):

    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    #Starting from |+> states
    
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[3])
    
    #Layer - 1

    
    circuit.rzz(beta[0], qreg_q[0], qreg_q[1])
    circuit.rzz(beta[0], qreg_q[2], qreg_q[3])
    circuit.rzz(beta[0], qreg_q[0], qreg_q[3])
    circuit.rx(gamma[0], qreg_q[0])
    circuit.rzz(beta[0], qreg_q[1], qreg_q[2])
    circuit.rx(gamma[0], qreg_q[3])
    circuit.rx(gamma[0], qreg_q[1])
    circuit.rx(gamma[0], qreg_q[2])
    
    #Layer - 2
    
    circuit.rzz(beta[1], qreg_q[0], qreg_q[1])
    circuit.rzz(beta[1], qreg_q[2], qreg_q[3])
    circuit.rzz(beta[1], qreg_q[0], qreg_q[3])
    circuit.rx(gamma[1], qreg_q[0])
    circuit.rzz(beta[1], qreg_q[1], qreg_q[2])
    circuit.rx(gamma[1], qreg_q[1])
    circuit.rx(gamma[1], qreg_q[2])
    circuit.rx(gamma[1], qreg_q[3])

    #Visualization - off
    #editor = CircuitComposer(circuit=circuit)
    #editor
    
    return circuit

#%% VQE

counts = []
values = []

def store_int_results(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)
    
#%%
    
vqe = VQE(H,circ,optimizer = SLSQP(maxiter=60),callback = store_int_results, gradient=Gradient(grad_method = 'param_shift'), quantum_instance = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator')))

result = vqe.compute_minimum_eigenvalue(operator = H)


#%%

IBMQ.disable_account()