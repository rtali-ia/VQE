#%%

from sqlalchemy import create_engine  
from sqlalchemy import Table, Column, String, MetaData

#%%


db_string = "postgres://admin:donotusethispassword@aws-us-east-1-portal.19.dblayer.com:15813/compose"
db = create_engine(db_string)

meta = MetaData(db)  

#engine = db.create_engine('dialect+driver://postgres:root@localhost:5432/db')