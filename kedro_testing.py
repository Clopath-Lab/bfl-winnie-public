#%%

import os
import sys
import pandas as pd


src_dir = os.path.join(os.getcwd(), '..', 'src')
sys.path.append(src_dir)

%run '/home/klara/bfl-winnie/.ipython/profile_default/startup/00-kedro-init.py'
%reload_kedro

#%%

from winnie3.d05_reporting.run_winnie import run_winnie

#%%
run_winnie(case_id=60)

# %%
rec = context.catalog.load('recommendations')

#%%