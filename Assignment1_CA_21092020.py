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

# I defined a function to plot the correlation plot for all continuous features.
df_corr = pass_model[['X_start', 'Y_start', 'C_start', 
        'X_end', 'Y_end', 'C_end', 'Distance_to_keep_start', 
        'Distance_to_keep_end', 'Distance_pass', 'dX', 'd_Distance']]

def corrplot(input_df):
    corr = input_df.corr()
    fig, ax = plt.subplots(figsize=(10,10)) 
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

corrplot(df_corr)

# Creating dummy variables here.
df_temp = pd.get_dummies(pass_model['subEventName'], prefix_sep='_', drop_first=True)
pass_model = pd.concat([pass_model, df_temp], axis=1)
pass_model = pass_model.drop(columns= ['subEventName'])
pass_model.columns=['Acc_pass', 'X_start', 'Y_start', 'C_start', 'X_end', 'Y_end', 'C_end', 'Distance_to_keep_start', 'Distance_to_keep_end', 'Distance_pass', 'dX', 'd_Distance', 'Hand_pass', 'Head_pass', 'High_pass', 'Launch', 'Simple_pass', 'Smart_pass']

# To check what is the average pass success rate. I will use it as a benchmark during classification.
pass_model['Acc_pass'].mean()

# Building a model on X_start and Y_start
pass_model_xystart = smf.glm(formula="Acc_pass ~ X_start + Y_start" , data=pass_model, 
                           family=sm.families.Binomial()).fit()
print(pass_model_xystart.summary())  

## - I gave a try to a number of quadratic model here.  ## A new dataset is created for this purpose.

pass_model_quad = pass_model[['Acc_pass', 'X_start', 'Y_start', 'Distance_to_keep_start', 'd_Distance', 'Head_pass', 'High_pass', 'Simple_pass']]
pass_model_quad['XY'] = pass_model_quad['X_start'] * pass_model_quad['Y_start']
pass_model_quad['X2'] = pass_model_quad['X_start']**2
pass_model_quad['Y2'] = pass_model_quad['Y_start']**2
pass_model_quad['Distance_to_keep_start2'] = pass_model_quad['Distance_to_keep_start']**2

# I have tried a number of quadratic models here. Below is only the last one that I gave a try. -
pass_model_xy_quad = smf.glm(formula="Acc_pass ~ Distance_to_keep_start2 + d_Distance + Head_pass + High_pass + Simple_pass" , data=pass_model_quad, 
                           family=sm.families.Binomial()).fit()
print(pass_model_xy_quad.summary())  

# Poor results gathered from the above model. Ideal model in my opinion is in the below. (No train/test split.) - Conclusion
# It has roughly -18.000 log likelyhood. Much higher than the quadratic model.
# I ran different models including all possible variables, but none of them leads to a significant increase in the log likehood.
# Which is why, I conclude with the simplest one, which is the one in the below.
# Simpler model is selected due to bias variance tradeoff. 

pass_model1 = smf.glm(formula="Acc_pass ~ Distance_to_keep_start + d_Distance + Head_pass + High_pass + Simple_pass" , data=pass_model, 
                           family=sm.families.Binomial()).fit()
print(pass_model1.summary()) 

# Below code is written to print out the summary table.
plt.rc('figure', figsize=(6, 3.5))
plt.text(0.01, 0.05, str(pass_model1.summary()), {'fontsize': 10}, fontproperties = 'monospace')
plt.axis('off')
plt.tight_layout()

# I concluded with that model so I added the probabilities as a column to the main dataframe.
pass_model['xP'] = pass_model1.predict(pass_model)
 

## The codes below here onwards are just to check the accuracy of the model quickly.
## - I made this as I think what if this was a classification problem.
## - I assume, a fair classification model would lead me to a fair xP model. 
   
# The dataset is separated into target (Acc_pass) and features. This will be used during classification.
y = pass_model['Acc_pass']
X = pass_model.drop(columns=['Acc_pass'])

# Split into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Logistic regression - for classification purposes. To check the accuracy of the model.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)

THRESHOLD = 0.80 ## Relying on the average pass accuracy. 
preds = np.where(clf.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)

pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds), roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"])

## - Classification accuracy is fair enough. Therefore I conclude this would be a fair model to use to calculate xP.

# I start with adding player names and role names to the main dataframe.
playerId = passes['playerId']
pass_model = pass_model.join(playerId)

# IDs must match with those in players data. 
with open('Wyscout/players.json') as name_file:
    data_id = json.load(name_file)
id_dataframe = pd.DataFrame(data_id)
id_dataframe = id_dataframe[['wyId', 'firstName', 'lastName']]
id_dataframe['Name'] = id_dataframe['firstName'].str.cat(id_dataframe['lastName'],sep=" ")
id_dataframe = id_dataframe.drop(columns = ['firstName', 'lastName'])
id_dataframe.columns=['playerId', 'Name']

df_final = pass_model.merge(id_dataframe, how='left', left_on='playerId', right_on='playerId')

## Position names are added to the main dataset as well. 
position_dataframe = pd.DataFrame(data_id)
position_dataframe['role_name'] = position_dataframe.role.apply(lambda x: x.get('name'))

position_dataframe = position_dataframe[['wyId', 'role_name']]
position_dataframe.columns=['playerId', 'role_name']

df_final = df_final.merge(position_dataframe, how='left', left_on='playerId', right_on='playerId')
## - Midfielders to be filtered out here - ##
df_midfielders = df_final[df_final['role_name']=='Midfielder']
df_xP = df_midfielders.groupby('Name')['xP', 'Acc_pass'].sum()
df_xP['xP_index'] = (df_xP['Acc_pass'] / df_xP['xP']) * 100
df_xP = df_xP[df_xP['Acc_pass'] >= 30]
df_xP = df_xP.sort_values(by ='xP_index', ascending=False)
df_average_acc = df_midfielders.groupby('Name')['Acc_pass'].mean()
df_xP = df_xP.merge(df_average_acc, how='left', left_on='Name', right_on='Name')

# I just added an extra filter to see so-called "Elite" passers (Among midfielders.)
df_final_analysis_elite = df_xP[['Acc_pass_x', 'xP_index', 'Acc_pass_y']]
df_final_analysis_elite = df_final_analysis_elite[df_final_analysis_elite['xP_index'] > 100]
df_final_analysis_elite = df_final_analysis_elite[df_final_analysis_elite['Acc_pass_y'] > 0.85]
df_final_analysis_elite = df_final_analysis_elite[df_final_analysis_elite['Acc_pass_x'] > 200]

# Some scouting after elite players are listed. Measures are relaxed a bit to see the potential elites. 
df_final_analysis = df_xP[['xP_index', 'Acc_pass_y']]

sns.scatterplot(data=df_final_analysis, x="Acc_pass_y", y="xP_index")

df_final_analysis2 = df_final_analysis[df_final_analysis['xP_index'] > 100]
df_final_analysis2 = df_final_analysis2[df_final_analysis2['Acc_pass_y'] > 0.85]
df_final_analysis2 = df_final_analysis2.sort_values(by ='xP_index', ascending = False)

## Scatter plot to point out some potentials.
figs, axes = plt.subplots(figsize=(12, 7))
axes = sns.regplot(data=df_final_analysis2, x="Acc_pass_y", y="xP_index", ci=None)
plt.text(0.96,111.5,'Daniel Amartey')
plt.text(0.89,107,'Jack Wilshere')
plt.text(0.907,106.7,'Steven Defour')
plt.text(0.896,105.6,'Granit Xhaka')
plt.text(0.867,105.5,'Matty James')
plt.text(0.88,105.15,'Jordan Henderson')
plt.text(0.89,104.5,'David Silva')
plt.text(0.864,103.96,'Andy King')
plt.text(0.8425,103.8,'Danny Drinkwater')
plt.text(0.8545,103.35,'Christian Eriksen')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
plt.xlabel('Pass Accuracy (Pass success rate)')
plt.ylabel('xP Index')


