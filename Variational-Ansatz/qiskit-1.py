# -*- coding: utf-8 -*-
"""
Implementing VQA Ansatz
VQE Full Function Definition - https://github.com/Qiskit/qiskit-aqua/blob/master/qiskit/aqua/algorithms/minimum_eigen_solvers/vqe.py#L274 

"""

#%%

TOKEN = 'f634b2a31452a1ee1d3693684e2a09e0bf30b4a581e90417a04c1cea9c356974562c7b45e945f864c9998376e56ee8c52f243c2360e91c2d4124157456ab4257'

#%%

import numpy as np
from numpy import pi
# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.tools.jupyter import *
from qiskit.visualization import *
#from ibm_quantum_widgets import *
from qiskit.aqua.utils import *
#from qiskit.aqua.operators import Z,I,X,Y
from qiskit.opflow import Z, X, I  # Pauli Z, X matrices and identity
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.algorithms.optimizers import SPSA,SLSQP
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.operators.gradients import Gradient, NaturalGradient, QFI, Hessian
from qiskit.aqua import QuantumInstance

#optimizer=SPSA()

#%%

# Loading your IBM Quantum account(s)
#provider = IBMQ.enable_account(TOKEN=TOKEN)


#%%

#Define the Hamiltonian
'''
I  = np.array([[ 1, 0],[ 0, 1]])
X = np.array([[ 0, 1],[ 1, 0]])
Y = np.array([[ 0,-1j],[1j, 0]])
Z = np.array([[ 1, 0],[ 0,-1]])
'''

h = 0.5
'''
H = -tensorproduct(Z,Z,I,I) - tensorproduct(I,Z,Z,I) - tensorproduct(I,I,Z,Z)
H = H - h*(tensorproduct(X,I,I,I) + tensorproduct(I,X,I,I) + tensorproduct(I,I,X,I) + tensorproduct(I,I,I,X))
'''

H = -(Z ^ Z ^ I ^ I) - (I ^ Z ^ Z ^ I) - (I ^ I ^ Z ^ Z) - h * ((X ^ I ^ I ^ I) + (I ^ X ^ I ^ I) + (I ^ I ^ X ^ I) + (I ^ I ^ I ^ X))

#%%

#Direct Diagonalization

from numpy import linalg as LA
w, v = LA.eig(H)
print(w)
print(v)


#%%
q0 = np.array([[1],[0]])
i = tensorproduct(q0,q0,q0,q0)

#%%

#Define Params for the circuit
params = [Parameter(r'$\beta_1$'), Parameter(r'$\beta_2$'), Parameter(r'$\gamma_1$'), Parameter(r'$\gamma_2$')]

#Define the Circuit
def circuit_var(params):
    
    # Params order beta1, beta2, gamma1, gamma2

    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    #Starting from |+> states
    
    circuit.h(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[3])
    
    #Layer - 1

    
    circuit.rzz(params[0], qreg_q[0], qreg_q[1])
    circuit.rzz(params[0], qreg_q[2], qreg_q[3])
    circuit.rzz(params[0], qreg_q[0], qreg_q[3])
    circuit.rx(params[2], qreg_q[0])
    circuit.rzz(params[0], qreg_q[1], qreg_q[2])
    circuit.rx(params[2], qreg_q[3])
    circuit.rx(params[2], qreg_q[1])
    circuit.rx(params[2], qreg_q[2])
    
    #Layer - 2
    
    circuit.rzz(params[1], qreg_q[0], qreg_q[1])
    circuit.rzz(params[1], qreg_q[2], qreg_q[3])
    circuit.rzz(params[1], qreg_q[0], qreg_q[3])
    circuit.rx(params[3], qreg_q[0])
    circuit.rzz(params[1], qreg_q[1], qreg_q[2])
    circuit.rx(params[3], qreg_q[1])
    circuit.rx(params[3], qreg_q[2])
    circuit.rx(params[3], qreg_q[3])

    #Visualization - off
    #editor = CircuitComposer(circuit=circuit)
    #editor
    
    return circuit

#%%

#Create a Variational_form to be passed to the VQE function.

class my_var_form(VariationalForm):
    def __init__(self, numpar, numq) -> None:
        super().__init__()
        self._num_parameters = numpar
        self._num_qubits = numq
        pass
    def construct_circuit(self,params):
        circuit = circuit_var(params)
        return circuit

circ = my_var_form(numpar=4,numq=4)
my_circ = circ.construct_circuit(params=params) # This Quantum circuit is passed to the VQE function.

#%% VQE

counts = []
values = []
#This call allows the optimizer to return values from inside the optizer while the optimization is on.
def store_int_results(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)
    
#%%

#all_zero = i #Created above

#init_angles = np.array([pi/3,pi/3,pi/3,pi/3])

#params = [Parameter(r'$\beta_1$'), Parameter(r'$\beta_2$'), Parameter(r'$\gamma_1$'), Parameter(r'$\gamma_2$')]

#circ = circuit_var()


#%% 

back_end = Aer.get_backend("statevector_simulator") #Initiate Backend

#Call the inbuilt VQE function.
vqe = VQE(H,var_form = my_circ,optimizer = SLSQP(maxiter=60),callback = store_int_results, gradient=Gradient(grad_method = 'param_shift'), quantum_instance = QuantumInstance(backend=back_end))

#Check the final result.
result = vqe.compute_minimum_eigenvalue(operator = H)


#%%

#IBMQ.disable_account()