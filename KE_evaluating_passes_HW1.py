import pandas as pd
import numpy as np
import json
#import matplotlib.pyplot as plt
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import sys
# import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from tqdm import tqdm
# from sklearn.preprocessing import PolynomialFeatures

with open('/Wyscout/events/events_England.json') as f:
    data = json.load(f)    
raw = pd.DataFrame(data)

passes = raw[raw['eventName']=='Pass']
passes = passes.sample(n = 50000, random_state = 42)

cols = ['Acc_pass', 'X_start', 'Y_start', 'C_start', 
        'X_end', 'Y_end', 'C_end', 'Distance_to_keep_start', 
        'Distance_to_keep_end', 'Distance_pass', 'match_period', "speed", "X_start_squared", "Y_start_squared", "C_start_squared", "X_end_squared", "Y_end_squared", "C_end_squared" ,"playerId"]
pass_model = pd.DataFrame(columns=cols)    

# Created only for exploration purposes.
passes_temp = passes.head(50000) # 50000
passes_temp['positions']

# Creating & preparing the dataset
for i,pass_ in tqdm(passes.iterrows()):
    
    # Pass start location
    pass_model.at[i,'X_start']=100-pass_['positions'][0]['x']
    pass_model.at[i,'Y_start']=pass_['positions'][0]['y']
    pass_model.at[i,'C_start']=abs(pass_['positions'][0]['y']-50)
    
    #Pass end location
    pass_model.at[i,'X_end']=100-pass_['positions'][1]['x']
    pass_model.at[i,'Y_end']=pass_['positions'][1]['y']
    pass_model.at[i,'C_end']=abs(pass_['positions'][1]['y']-50)
    pass_model.at[i, "player_id"] = pass_["playerId"]
    
    ## squared 
    pass_model.at[i,'X_start_squared']=(100-pass_['positions'][0]['x'])**2
    pass_model.at[i,'Y_start_squared']=(pass_['positions'][0]['y'])**2
    pass_model.at[i,'C_start_squared']=(abs(pass_['positions'][0]['y']-50))**2

    pass_model.at[i,'X*Y_start_squared']=(100-pass_['positions'][0]['x']) * (pass_['positions'][0]['y'])
    
    #Pass end location
    pass_model.at[i,'X_end_squared']=(100-pass_['positions'][1]['x'])**2
    pass_model.at[i,'Y_end_squared']=(pass_['positions'][1]['y'])**2
    pass_model.at[i,'C_end_squared']=(abs(pass_['positions'][1]['y']-50))**2

    pass_model.at[i,'X*Y_end_squared']= (100-pass_['positions'][1]['x']) * (pass_['positions'][1]['y'])
    ## end squared
    
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
    
    pass_model.at[i, "timeelapsed"] = pass_["eventSec"]
    pass_model.at[i, "match_period"] = pass_["matchPeriod"]
        
    #Accurate passes   
    pass_model.at[i,'Acc_pass']=0
    for passtags in pass_['tags']:
        if passtags['id']==1801:
            pass_model.at[i,'Acc_pass']=1
    sys.stdout.write('.'); sys.stdout.flush(); #Just for visual check while the code is runnung, whether the loop works or not.

# I added to more columns, which may come in handy while modelling. 
pass_model['dX'] = pass_model['X_start'] - pass_model['X_end']
pass_model['d_Distance'] = pass_model['Distance_to_keep_start'] - pass_model['Distance_to_keep_end']
pass_model["match_period"] = pass_model["match_period"].astype("category").cat.codes
pass_model = pass_model.astype(float)

pass_model['speed'] = pass_model["dX"] / pass_model["timeelapsed"]
# Turn them into floats as correlations cannot be plotted while all columns are objects.
# Pass types are added from raw data.
subEventName = passes['subEventName']
pass_model = pass_model.join(subEventName)
pass_model["subEventName"] = pass_model["subEventName"].astype("category").cat.codes

#condition: last third passes
pass_model = pass_model[pass_model['Distance_to_keep_end'] < 33]

#preparing data for training
pass_model2train = pass_model.drop(["speed", "timeelapsed"], axis=1)
x_axis_dataframe = pass_model2train.drop(["Acc_pass", "player_id", "playerId"], axis=1)
y_axis_dataframe = pass_model2train["Acc_pass"]
# to list in order preparing the input for sklearn
x_axis_features = x_axis_dataframe.values.tolist() 
#normalisation
scaler = preprocessing.MinMaxScaler()
scaler = scaler.fit(x_axis_features)
x_axis_features = scaler.transform(x_axis_features)
# to list in order preparing the input for sklearn
y_axis_labels = y_axis_dataframe.values.tolist()

#linear regression training
X_train, X_test, y_train, y_test = train_test_split(x_axis_features, y_axis_labels, test_size=0.33)
clf = LogisticRegression(random_state=0, max_iter=10000, tol=0.000001, verbose=1).fit(X_train, y_train) 

y_predictions = clf.predict(X_test)
y_prediction_labels = list(zip(y_predictions, y_test))
pred_correctness = [x[0] == x[1] for x in y_prediction_labels]
accuracy_now = pred_correctness.count(True) / len(pred_correctness)
# accuracy test blind
accuracy_now 

# testing on the training set to check if the model complexity needs to be increased
y_predictions_train = clf.predict(X_train)
y_prediction_labels_train = list(zip(y_predictions_train, y_train))
pred_correctness_train = [x[0] == x[1] for x in y_prediction_labels_train]
accuracy_now_train = pred_correctness_train.count(True) / len(pred_correctness_train)
accuracy_now_train 

## Checked train and test accuracy -> found out that the train accuracy is 85%, so there is room for increasing the model's complexity -> I hearby keep checking the accuracy on test to make sure we are not falling into the curse of dimensionality aka overfitting

# by normalizing the data 0-1 improved gradient descent and convergence

# all features polynomial degree 2

#extracting the coefficients
list_of_coeffs_INIT = clf.coef_
list_of_coeffs = [(x, y) for x,y in enumerate(list_of_coeffs_INIT[0])]
list_of_coeffs2 = [(x, y) for x,y in enumerate(list_of_coeffs_INIT[0])]
list_of_coeffs.sort(key= lambda x: x[1])
list_of_coeffs2.sort(key= lambda x: x[1], reverse=True)

# negative coeffs
list_of_coeffs[:10]

# positive coeffs
list_of_coeffs2[:10]

#corresponding features to those coeffs
x_axis_dataframe.columns

## Retrieving list of best player, worst players in the sample taken as well as their score according to my model

pass_model_results = pass_model2train.drop(["playerId"], axis=1)
distinct_player_ids = pass_model2train["player_id"].unique()

#processing the IDs and their respective score in the model
player_ids_to_scores = {}
player_ids_to_scores_proba = {}
player_ids_to_no_passes = {}
player_ids_with_diffs = []
for player_id in tqdm(distinct_player_ids):
    temp_df = pass_model2train[pass_model2train["player_id"] == player_id]
    temp_df = temp_df.drop(["player_id", "Acc_pass", "playerId"], axis=1)
    temp_vals = temp_df.values.tolist()
    player_ids_to_no_passes[int(player_id)] = len(temp_vals)
    temp_vals_scaled = scaler.transform(temp_vals)
    y_predictions = clf.predict(temp_vals_scaled)
    y_predictions_average = np.mean(y_predictions)
    y_predictions_proba = clf.predict_proba(temp_vals_scaled)
    y_predictions_average_proba = np.mean([x[1] for x in y_predictions_proba])
    player_ids_to_scores[int(player_id)] = y_predictions_average
    player_ids_to_scores_proba[int(player_id)] = y_predictions_average_proba
    if (y_predictions_average_proba >= 0.5) != (y_predictions_average >= 0.5):
        player_ids_with_diffs.append(player_id)

    
    #processing the mapping (id <--> player Name)
with open('Wyscout/players.json') as f:
    playersdata = json.load(f)
player_name_to_id = {}
player_id_to_name = {}
all_player_data_w_id = {}
for data_entry in playersdata:
    player_name_to_id[data_entry["shortName"]] = data_entry["wyId"]
    all_player_data_w_id[data_entry["wyId"]] = data_entry
    player_id_to_name[data_entry["wyId"]] = data_entry["shortName"]

#Retrieve best and worst players and their model score
id_to_score_tuple_list = [(x, y) for x,y in player_ids_to_scores_proba.items()]
id_to_score_tuple_list2 = [(x, y) for x,y in player_ids_to_scores_proba.items()]

#Retrieve best 10 and worst 10 players and their model score
id_to_score_tuple_list.sort(key=lambda x:x[1], reverse=False)
id_to_score_tuple_list2.sort(key=lambda x:x[1], reverse=True)
number_wanted = 10
#id_to_score_tuple_list[:number_wanted] # top 10 worst players in passing (id, score)
#id_to_score_tuple_list2[:number_wanted] # top 10 best players in passing (id, score)

#Top 10 --> top 10 best players in passing (id, score)
top_players_info = [(rank+1, player_id_to_name[x[0]], x[0], x[1]) for rank, x in enumerate(id_to_score_tuple_list2[:number_wanted])] 
fbest = open('bestPlayers.txt', 'w')
for listing in top_players_info:
    print(listing)
    fbest.write("".join(str(listing)))
    fbest.write("\n")
fbest.close()

worst_players_info = [(rank+1, player_id_to_name[x[0]], x[0], x[1]) for rank, x in enumerate(id_to_score_tuple_list[:number_wanted])] # top 10 best players in passing (id, score)
#worst_players_names = []
fworst = open('worstPlayers.txt', 'w')
for listing in worst_players_info:
    print(listing)
    fworst.write("".join(str(listing)))
    fworst.write("\ n")
    #worst_players_names = str(listing) + '\n'
fworst.close()

#check some players
def retrievePlayerDetails(wanted_player_inquiry):
    print(player_id_to_name[wanted_player_inquiry])
    return all_player_data_w_id[wanted_player_inquiry]
wanted_player_inquiry = 14886
print(player_id_to_name[wanted_player_inquiry])
all_player_data_w_id[wanted_player_inquiry]


