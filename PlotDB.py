#%%
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

class Entry(base):
    __table_args__ = {'schema' : 'Logs', 'extend_existing': True}
    __tablename__ = 'iterlog'
    logid = Column(Integer, primary_key=True)
    owner = Column(String)
    n_qubits = Column(Integer)
    g = Column(Numeric)
    layers = Column(Integer)
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
    round = Column(Integer)
    overlap = Column(Numeric)
    energy = Column(Numeric)
    ansatz = Column(String)
    params = Column(String)
    log_start_time = Column(TIMESTAMP)
    atype = Column(String)

#%%
#Format for specifying log_start_time = '2021-10-14 00:00:00'

def fetch_for_plotting_round(atype,n_qubits,g,log_start_time):
  table_df = pd.read_sql_table('roundlog', con=db, schema='Logs')
  
  try:
      subset_df = table_df[(table_df.atype == atype) & (table_df.n_qubits == n_qubits) & (table_df.log_start_time >= '2021-10-14 00:00:00')]
      return subset_df
  except Exception as e:
      return str(e)
  


#%%
dset = fetch_for_plotting_round('hw',6,0.5,'2021-10-14 00:00:00')

plt.plot(dset.iloc[:,5],dset.overlap)
plt.title('Track Cost Function')
plt.xlabel('Number of Iterations')
plt.ylabel('Minimum Eigen Value attained')
plt.show()

# %%
