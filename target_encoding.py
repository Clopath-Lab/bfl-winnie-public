#%%
# load kedro context
%run '/home/klara/bfl-winnie/.ipython/profile_default/startup/00-kedro-init.py'
%reload_kedro

# %%

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import yaml
import os

messages_table = context.catalog.load("primary_messages")  

setup_dir = '/home/klara/bfl-winnie'
credentials_path = os.path.join(setup_dir, 'conf', 'local', 'credentials.yml')
with open(credentials_path, 'r') as credential_file:
    credentials = yaml.safe_load(credential_file)
    
parameter_path = os.path.join(setup_dir, 'conf', 'base', 'parameters.yml')
with open(parameter_path, 'r') as parameter_file:
    parameters = yaml.safe_load(parameter_file)

db_host = parameters['DATABASE_PARAMS']['db_host']
db_name = parameters['DATABASE_PARAMS']['db_name']
db_user = credentials['dssg']['username']
db_pass = credentials['dssg']['password']

conn = create_engine('mysql+pymysql://%s:%s@%s/%s' % 
                       (db_user, db_pass, db_host, db_name),
                       encoding = 'latin1', 
                       echo = True)


#%%

query = "case_issues" 
case_issues = pd.read_sql(query, conn, params=("%<br/>%",))
case_issues['issue_id'] = case_issues['issue_id'].astype(str).astype(int)
case_issues['case_id'] = case_issues['case_id'].astype(str).astype(int)


query = "issues" 
issues = pd.read_sql(query, conn, params=("%<br/>%",))
issues.columns = ['issue_id', 'name', 'description', 'created_at', 'updated_at']


# TODO: standardise issue names, remove duplicates
issues['name'].nunique()
#issues['name'].str.lower().nunique()
issues.drop(['created_at', 'updated_at','description'], axis=1, inplace=True)


case_issues = case_issues.merge(issues)


#%%
'''
query = "case_lead_lawyer" 
case_lead_lawyer = pd.read_sql(query, conn, params=("%<br/>%",))

case_lead_lawyer = case_lead_lawyer.drop_duplicates()

case_issues = case_issues.merge(case_lead_lawyer)
'''
#%%

# take only messages with case_ids

train_data = messages_table[['case_id','id','question','answer']]
train_data = train_data.dropna() 
train_data.reset_index(drop=True, inplace=True)

# %%

train_data = train_data.merge(case_issues,how='inner')

train_data = pd.concat([train_data.drop('issue_id', 1), pd.get_dummies(train_data.issue_id).mul(1)], axis=1)

train_data_grouped = train_data.groupby(['case_id','id','question','answer']).sum()

train_data_grouped.reset_index(inplace=True)

columns = train_data_grouped.columns
train_data_grouped['n_issues'] = train_data_grouped[columns[4:]].sum(axis=1)

train_data_grouped.to_pickle('/datadrive/issues/data.pkl')

