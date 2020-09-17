import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import FCPython 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc,recall_score,precision_score


with open('Wyscout/events/events_England.json') as f:
    data = json.load(f)
    
raw = pd.DataFrame(data)
# Just to check what the columns are
for col in raw.columns: 
    print(col) 

passes = raw[raw['eventName']=='Pass']
# Due to computational reasons, I have to select a random subsample.
# I set the random_state = 42, in order to make sure that it will take out the same sample for each run.
passes = passes.sample(n = 50000, random_state = 42)
cols = ['Acc_pass', 'X_start', 'Y_start', 'C_start', 
        'X_end', 'Y_end', 'C_end', 'Distance_to_keep_start', 
        'Distance_to_keep_end', 'Distance_pass' ]
pass_model = pd.DataFrame(columns=cols)

# Created only for exploration purposes.
passes_temp = passes.head(200)

# Creating the dataset
for i,pass_ in passes.iterrows():
    
    # Pass start location
    pass_model.at[i,'X_start']=100-pass_['positions'][0]['x']
    pass_model.at[i,'Y_start']=pass_['positions'][0]['y']
    pass_model.at[i,'C_start']=abs(pass_['positions'][0]['y']-50)
    
    #Pass end location
    pass_model.at[i,'X_end']=100-pass_['positions'][1]['x']
    pass_model.at[i,'Y_end']=pass_['positions'][1]['y']
    pass_model.at[i,'C_end']=abs(pass_['positions'][1]['y']-50)
    
    #Distances (from the start location of the pass to the keep)
    x_start=pass_model.at[i,'X_start']*105/100
    y_start=pass_model.at[i,'C_start']*65/100
    pass_model.at[i,'Distance_to_keep_start']=np.sqrt(x_start**2 + y_start**2)
    
    #Distances (from the end location of the pass to the keep)
    x_end=pass_model.at[i,'X_end']*105/100
    y_end=pass_model.at[i,'C_end']*65/100
    pass_model.at[i,'Distance_to_keep_end']=np.sqrt(x_end**2 + y_end**2)
    
    #Pass distance
    pass_model.at[i,'Distance_pass'] = np.sqrt(abs(x_start-x_end)**2 + abs(y_start-y_end)**2)
        
    #Accurate passes   
    pass_model.at[i,'Acc_pass']=0
    for passtags in pass_['tags']:
        if passtags['id']==1801:
            pass_model.at[i,'Acc_pass']=1
    sys.stdout.write('.'); sys.stdout.flush(); #Just for visual check while the code is runnung, whether the loop works or not.

# I added to more columns, which may come in handy while modelling. 
pass_model['dX'] = pass_model['X_start'] - pass_model['X_end']
pass_model['d_Distance'] = pass_model['Distance_to_keep_start'] - pass_model['Distance_to_keep_end']

# Turn them into floats as correlations cannot be plotted while all columns are objects.
pass_model = pass_model.astype(float)
# Pass types are added from raw data.
subEventName = passes['subEventName']
pass_model = pass_model.join(subEventName)