# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import FCPython 

# Load event 
wyscout_events = 'Wyscout/events'
data = pd.read_json(wyscout_events+'/events_England.json')

# Define variables related to type and success of 
successful_pass_id = 1801
useful_passes = ['Simple pass', 'High pass', 'Smart pass', 'Cross']
# Filter out passes by hand and head
df = data[data['subEventName'].isin(useful_passes)]
passes_df = df[df.eventName=='Pass']

# Calculates angle from the goal given a position
def calculate_angle(x, y):
    a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
    if a<0:
        a=np.pi+a
    return a

# Curate the data and create various variables for the models
success_list = []
position_x_init = []
position_y_init = []
position_x_end = []
position_y_end = []
angle_init = []
angle_end = []
in_box = []
in_six_yard = []
for index, pass_event in passes_df.iterrows():
    position_x_init.append(pass_event['positions'][0]['x'])
    position_y_init.append(pass_event['positions'][0]['y'])
    position_x_end.append(pass_event['positions'][1]['x'])
    position_y_end.append(pass_event['positions'][1]['y'])
    angle_init.append(calculate_angle((100 - pass_event['positions'][0]['x'])*1.05, abs(50 - pass_event['positions'][0]['y'])*0.65))
    angle_end.append(calculate_angle((100 - pass_event['positions'][1]['x'])*1.05, abs(50 - pass_event['positions'][1]['y'])*0.65))
    
    # Create list with 0 and 1 for passes inside and outside the box
    if pass_event['positions'][1]['x'] >= 84 and pass_event['positions'][1]['y'] > 22 and pass_event['positions'][1]['y'] < 75:
        in_box.append(1)
    else:
        in_box.append(0)
    if pass_event['positions'][1]['x'] >= 94 and pass_event['positions'][1]['y'] > 37 and pass_event['positions'][1]['y'] < 63:
        in_six_yard.append(1)
    else:
        in_six_yard.append(0)
    
    # Create list with 0 and 1 for unsuccessful and successful passes
    if 1801 in [tag['id'] for tag in pass_event['tags']]:
        success_list.append(1)
    else:
        success_list.append(0)

# Add parameters to passes dataframe
passes_df['successful'] = success_list
passes_df['position_x_init_nm'] = position_x_init
passes_df['position_y_init_nm'] = position_y_init
passes_df['position_x_end_nm'] = position_x_end
passes_df['X'] = 100 - np.asarray(position_x_end)
passes_df['position_y_end_nm'] = position_y_end
passes_df['C'] = abs(np.asarray(position_y_end) - 50)
passes_df['position_x_init'] = np.asarray(position_x_init)*1.05
passes_df['position_y_init'] = np.asarray(position_y_init)*0.65
passes_df['position_x_end'] = np.asarray(position_x_end)*1.05
passes_df['position_y_end'] = np.asarray(position_y_end)*0.65
passes_df['position_xy_init'] = np.asarray(position_x_init)*1.05 * np.asarray(position_y_init)*0.65
passes_df['position_xy_end'] = np.asarray(position_x_end)*1.05 * np.asarray(position_y_end)*0.65
passes_df['distance_x_init'] = (100 - np.asarray(position_x_init))*105
passes_df['goal_distance'] = np.sqrt((105 - passes_df['position_x_init'])**2 + (32.5 - passes_df['position_y_init'])**2)
passes_df['goal_distance_end'] = np.sqrt((105 - passes_df['position_x_end'])**2 + (32.5 - passes_df['position_y_end'])**2)
passes_df['goal_distance_reduction'] = np.sqrt((105 - passes_df['position_x_end'])**2 + (32.5 - passes_df['position_y_end'])**2) - np.sqrt((105 - passes_df['position_x_init'])**2 + (32.5 - passes_df['position_y_init'])**2)
passes_df['pass_distance'] = np.sqrt((passes_df['position_x_init'] - passes_df['position_x_end'])**2 + (passes_df['position_y_init'] - passes_df['position_y_end'])**2) 
passes_df['angle_init'] = angle_init
passes_df['angle_end'] = angle_end
passes_df['in_box'] = in_box
passes_df['in_six_yard'] = in_six_yard
passes_df['distance_x_init'] = (100 - np.array(position_x_init))*1.05
passes_df['distance_y_init'] = (100 - np.array(position_y_init))*0.65
passes_df['distance_x_end'] = (100 - np.array(position_x_end))*1.05
passes_df['distance_y_end'] = (100 - np.array(position_y_end))*0.65
passes_df['distance_x_init_sq'] = ((100 - np.array(position_x_init))*1.05)**2
passes_df['distance_y_init_sq'] = ((100 - np.array(position_y_init))*0.65)**2
passes_df['distance_x_end_sq'] = ((100 - np.array(position_x_end))*1.05)**2
passes_df['distance_y_end_sq'] = ((100 - np.array(position_y_end))*0.65)**2

# Filter out passes that do not end up in the attacking half and passes that increase the distance from the goal
passes_df = passes_df[passes_df['position_x_end_nm'] >= 50]
passes_df = passes_df[passes_df['goal_distance_reduction'] > 0]
print('Total number of passes:', len(passes_df))

# Function that prints some basic statistics about the success rate of passes based on start and end position
# as well as the pass distance and whether it ends up in the six yard or the box
def basic_stats(passes_df):
    successful_total = len(passes_df[passes_df.successful==1].index)
    unsuccessful_total = len(passes_df[passes_df.successful==0].index)
    total = successful_total + unsuccessful_total
    successful_second = len(passes_df[(passes_df.successful==1) & (passes_df['position_x_init_nm']>=50)].index)
    unsuccessful_second = len(passes_df[(passes_df.successful==0) & (passes_df['position_x_init_nm']>=50)].index)
    total_second = successful_second + unsuccessful_second
    successful_third = len(passes_df[(passes_df.successful==1) & (passes_df['position_x_init_nm']>75)].index)
    unsuccessful_third = len(passes_df[(passes_df.successful==0) & (passes_df['position_x_init_nm']>75)].index)
    total_third = successful_third + unsuccessful_third
    successful_thirdp = len(passes_df[(passes_df.successful==1) & (passes_df['position_x_init_nm']>75) & (passes_df['position_x_init_nm'] < passes_df['position_x_end_nm'])].index)
    unsuccessful_thirdp = len(passes_df[(passes_df.successful==0) & (passes_df['position_x_init_nm']>75) & (passes_df['position_x_init_nm'] < passes_df['position_x_end_nm'])].index)
    total_thirdp = successful_thirdp + unsuccessful_thirdp

    successful_thirddist = len(passes_df[(passes_df.successful==1) & (passes_df['goal_distance'] < 20) & (passes_df['position_x_init_nm'] < passes_df['position_x_end_nm'])].index)
    unsuccessful_thirddist = len(passes_df[(passes_df.successful==0) & (passes_df['goal_distance'] < 20) & (passes_df['position_x_init_nm'] < passes_df['position_x_end_nm'])].index)
    total_thirddist = successful_thirddist + unsuccessful_thirddist
    
    successful_box = len(passes_df[(passes_df.successful==1) & (passes_df['in_box'] == 1)].index)
    unsuccessful_box = len(passes_df[(passes_df.successful==0) & (passes_df['in_box'] == 1)].index)
    total_box = successful_box + unsuccessful_box
    
    successful_six_yard = len(passes_df[(passes_df.successful==1) & (passes_df['in_six_yard'] == 1)].index)
    unsuccessful_six_yard = len(passes_df[(passes_df.successful==0) & (passes_df['in_six_yard'] == 1)].index)
    total_six_yard = successful_six_yard + unsuccessful_six_yard
    
    if total != 0:
        percentage_successful = successful_total / total
        print('Percentage total', percentage_successful)
    if total_second != 0:
        percentage_successful_second =  successful_second / total_second
        print('Percentage second', percentage_successful_second)
    if total_third != 0:
        percentage_successful_third =  successful_third / total_third
        print('Percentage third', percentage_successful_third)
    if total_thirdp != 0:
        percentage_successful_thirdp =  successful_thirdp / total_thirdp
        print('Percentage third and forward', percentage_successful_thirdp)
    if total_thirddist != 0:
        percentage_successful_thirddist = successful_thirddist / total_thirddist
        print('Distance smaller than 10 and forward', percentage_successful_thirddist)
    if total_box != 0:
        percentage_successful_box = successful_box / total_box
        print('Percentage in box', percentage_successful_box)
    if total_six_yard != 0:
        percentage_successful_six_yard = successful_six_yard / total_six_yard
        print('Percentage in 6 yard', percentage_successful_six_yard)

print('Printing basic statistics to get an idea about the data')
basic_stats(passes_df)

# Scatter plot for the 500 first passes and their success, based on the starting position
ax = passes_df.head(1000).plot.scatter(x='position_x_init', y='position_y_init',c=['green' if suc == 1 else 'red' for suc in passes_df['successful'].iloc[:200]])

model = passes_df.copy()

pitchLengthX = 105
pitchWidthY = 65

# Heat map showing the number of successful passes per area
H_Pass=np.histogram2d(model['position_y_end'], model['position_x_end'],bins=10,range=[[0, pitchWidthY],[0, pitchLengthX]])
Succ_Pass = np.histogram2d(model[model.successful==1]['position_y_end'], model[model.successful==1]['position_x_end'],bins=10,range=[[0, pitchWidthY],[0, pitchLengthX]])
(fig,ax) = FCPython.createPitch(pitchLengthX,pitchWidthY,'yards','gray')
pos=ax.imshow(Succ_Pass[0]/H_Pass[0], extent=[0,105,0,65], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Successful passes per ending position')
plt.xlim((-1,121))
plt.ylim((-3,83))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Number of successful passes in the attacking 40 meters of the pitch
successful_only=model[(model['successful']==1) & (model['X']<40)]
H_Pass=np.histogram2d(successful_only['X'], successful_only['position_y_end_nm'],bins=50,range=[[0, 100],[0, 100]])
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_Pass[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of successful passes')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Number of successful passes in box
successful_only=model[(model.in_box == 1) & (model['successful']==1)]
H_Goal=np.histogram2d(successful_only['X'], successful_only['position_y_end_nm'],bins=50,range=[[0, 100],[0, 100]])
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_Goal[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds)
fig.colorbar(pos, ax=ax)
ax.set_title('Number of passes')
plt.xlim((-1,66))
plt.ylim((-3,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Show how distance from goal changes the probability of successful pass
# In this case, the smaller the distance the smaller the chance of successful pass
passcount_dist=np.histogram(model['goal_distance_end'],bins=40,range=[0, 150])
spasscount_dist=np.histogram(model[model.successful==1]['goal_distance_end'],bins=40,range=[0, 150])
prob_pass=np.divide(spasscount_dist[0],passcount_dist[0])
dist=passcount_dist[1]
middist= (dist[:-1] + dist[1:])/2
fig,ax=plt.subplots(num=2)
ax.plot(middist, prob_pass, linestyle='none', marker= '.', markerSize= 12, color='black')
ax.set_ylabel('Probability of successful pass')
ax.set_xlabel("Pass distance from goal")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Definition of the model evaluation dataframe
col_names = [['model_name', 'tp', 'tn', 'fp', 'fn', 'total']]
models_eval = pd.DataFrame(columns = col_names)

# Make predictions for each pass based on the model
def calculate_xP(sh, model_variables,b):
    bsum=b[0]
    for i,v in enumerate(model_variables):
        bsum=bsum+b[i+1]*sh[v]
    xP = 1 - 1/(1+np.exp(bsum)) 
    return xP

# Add calculated data to the model evaluation dataframe
def add_to_dict(model_name, tp, tn, fp, fn, total, models_eval):
    tmp = pd.DataFrame({'model_name':[model_name], 'tp': [tp], 'tn': [tn], 'fp': [fp], 'fn': [fn], 'total': [total]}) 
    models_eval.loc[-1] = [model_name, tp, tn, fp, fn, total]
    models_eval.index = models_eval.index + 1
    models_eval = models_eval.sort_index() 

# Calculate confusion matrix and print the results
def confusion_matrix(model, model_name, models_eval, is_model):
    print(model_name)
    tp = len(model[(model.successful == 1) & (model[model_name] > 0.5)])
    tn = len(model[(model.successful == 0) & (model[model_name] < 0.5)])
    fp = len(model[(model.successful == 0) & (model[model_name] > 0.5)])
    fn = len(model[(model.successful == 1) & (model[model_name] < 0.5)])
    total = tp + tn + fp + fn
    print('True positive:', tp, 'percentage:', tp/total)
    print('True negative:', tn, 'percentage:', tn/total)
    print('False positive:', fp, 'percentage:', fp/total)
    print('False negative:', fn, 'percentage:', fn/total)
    print('Correct percentage:', tp/total + tn/total)
    if is_model:
        add_to_dict(model_name, tp, tn, fp, fn, total, models_eval)

# Help function for calculating expected pass and confusion matrix
def calculate_xP_and_confusion(model, model_vars, model_name, test_model, models_eval, is_model=True):
    b=test_model.params
    xP=calculate_xP(model, model_vars,b)
    model[model_name] = xP
    confusion_matrix(model, model_name, models_eval, is_model)

# Run different models and add their statistics to the model evaluation dataframe

# Starting position (x)
test_model_x = smf.glm(formula="successful ~ position_x_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_x.summary())

model_variables_x = ['position_x_init']
model_name_x = 'test_model_x'
calculate_xP_and_confusion(model, model_variables_x, model_name_x, test_model_x, models_eval)

# Starting position (x,y)
test_model_xy = smf.glm(formula="successful ~ position_x_init + position_y_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy.summary())

model_variables_xy = ['position_x_init', 'position_y_init']
model_name_xy = 'test_model_xy'
calculate_xP_and_confusion(model, model_variables_xy, model_name_xy, test_model_xy, models_eval)

# Distance in x axes
test_model_x_dist = smf.glm(formula="successful ~ distance_x_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_x_dist.summary())

model_variables_x_dist = ['distance_x_init']
model_name_x_dist = 'test_model_x_dist'
calculate_xP_and_confusion(model, model_variables_x_dist, model_name_x_dist, test_model_x_dist, models_eval)


# Distance of starting point in x and y axes
test_model_xy_dist = smf.glm(formula="successful ~ distance_x_init + distance_y_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_dist.summary())

model_variables_xy_dist = ['distance_x_init', 'distance_y_init']
model_name_xy_dist = 'test_model_xy_dist'
calculate_xP_and_confusion(model, model_variables_xy_dist, model_name_xy_dist, test_model_xy_dist, models_eval)

# Distance of ending point in x and y axes
test_model_xy_end_dist = smf.glm(formula="successful ~ distance_x_end + distance_y_end" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_end_dist.summary())

model_variables_xy_end_dist = ['distance_x_end', 'distance_y_end']
model_name_xy_end_dist = 'test_model_xy_end_dist'
calculate_xP_and_confusion(model, model_variables_xy_end_dist, model_name_xy_end_dist, test_model_xy_end_dist, models_eval)


# Starting position squared (x)
test_model_x_sq = smf.glm(formula="successful ~ distance_x_init_sq" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_x_sq.summary())

model_variables_x_sq = ['distance_x_init_sq']
model_name_x_sq = 'test_model_x_sq'
calculate_xP_and_confusion(model, model_variables_x_sq, model_name_x_sq, test_model_x_sq, models_eval)


# Starting position squared in x and y axes
test_model_xy_sq = smf.glm(formula="successful ~ distance_x_init_sq + distance_y_init_sq" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_sq.summary())

model_variables_xy_sq = ['distance_x_init_sq', 'distance_y_init_sq']
model_name_xy_sq = 'test_model_xy_sq'
calculate_xP_and_confusion(model, model_variables_xy_sq, model_name_xy_sq, test_model_xy_sq, models_eval)


# Starting position x*y
test_model_xy_mul = smf.glm(formula="successful ~ position_xy_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_mul.summary())

model_variables_xy_mul = ['position_xy_init']
model_name_xy_mul = 'test_model_xy_mul'
calculate_xP_and_confusion(model, model_variables_xy_mul, model_name_xy_mul, test_model_xy_mul, models_eval)


# Ending position x*y
test_model_xy_mul_end = smf.glm(formula="successful ~ position_xy_end" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_mul_end.summary())

model_variables_xy_mul_end = ['position_xy_end']
model_name_xy_mul_end = 'test_model_xy_mul_end'
calculate_xP_and_confusion(model, model_variables_xy_mul_end, model_name_xy_mul_end, test_model_xy_mul_end, models_eval)


# Starting and ending position x*y
test_model_xy_mul_both = smf.glm(formula="successful ~ position_xy_init + position_xy_end" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xy_mul_both.summary())

model_variables_xy_mul_both = ['position_xy_init', 'position_xy_end']
model_name_xy_mul_both = 'test_model_xy_mul_both'
calculate_xP_and_confusion(model, model_variables_xy_mul_both, model_name_xy_mul_both, test_model_xy_mul_both, models_eval)


# Starting position in x and y axes and angle
test_model_xya_init = smf.glm(formula="successful ~ distance_x_init_sq + distance_y_init_sq + angle_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xya_init.summary())

model_variables_xya_init = ['distance_x_init_sq', 'distance_y_init_sq', 'angle_init']
model_name_xya_init = 'test_model_xya_init'
calculate_xP_and_confusion(model, model_variables_xya_init, model_name_xya_init, test_model_xya_init, models_eval)


# Ending position in x and y axes and angle
test_model_xya_end = smf.glm(formula="successful ~ distance_x_end_sq + distance_y_end_sq + angle_end + in_box" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xya_end.summary())

model_variables_xya_end = ['distance_x_end_sq', 'distance_y_end_sq', 'angle_end', 'in_box']
model_name_xya_end = 'test_model_xya_end'
calculate_xP_and_confusion(model, model_variables_xya_end, model_name_xya_end, test_model_xya_end, models_eval)


# Ending position in x and y axes and angle
test_model_xyab_end = smf.glm(formula="successful ~ distance_x_end_sq + distance_y_end_sq + angle_end" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_xyab_end.summary())

model_variables_xyab_end = ['distance_x_end_sq', 'distance_y_end_sq', 'angle_end']
model_name_xyab_end = 'test_model_xyab_end'
calculate_xP_and_confusion(model, model_variables_xyab_end, model_name_xyab_end, test_model_xyab_end, models_eval)


# ALL INCLUDED
test_model_all = smf.glm(formula="successful ~ position_x_init_nm + position_y_init_nm + position_x_end_nm + position_y_end_nm + position_x_init + position_y_init + position_x_end + position_y_end + goal_distance + goal_distance_end + pass_distance + angle_init + angle_end + distance_x_init + distance_y_init + distance_x_end + distance_y_end + distance_x_init_sq + distance_y_init_sq + distance_x_end_sq + distance_y_end_sq" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_all.summary())

model_variables_all = ['position_x_init_nm', 'position_y_init_nm', 'position_x_end_nm', 'position_y_end_nm', 'position_x_init', 'position_y_init', 'position_x_end', 'position_y_end', 'goal_distance', 'goal_distance_end', 'pass_distance', 'angle_init', 'angle_end', 'distance_x_init', 'distance_y_init', 'distance_x_end', 'distance_y_end', 'distance_x_init_sq', 'distance_y_init_sq', 'distance_x_end_sq', 'distance_y_end_sq']
model_name_all = 'test_model_all'
calculate_xP_and_confusion(model, model_variables_all, model_name_all, test_model_all, models_eval)


# Goal distance reduction
test_model_gdr = smf.glm(formula="successful ~ goal_distance_reduction + angle_end + position_x_init + position_y_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_gdr.summary())

model_variables_gdr = ['goal_distance_reduction', 'distance_x_end_sq', 'position_x_init', 'position_y_init']
model_name_gdr = 'test_model_gdr'
calculate_xP_and_confusion(model, model_variables_gdr, model_name_gdr, test_model_gdr, models_eval)

# Goal distance reduction
test_model_gdr1 = smf.glm(formula="successful ~ goal_distance_end + angle_end + position_x_init + goal_distance_reduction" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_gdr1.summary())

model_variables_gdr1 = ['goal_distance_end', 'angle_end', 'position_x_init', 'goal_distance_reduction']
model_name_gdr1 = 'test_model_gdr1'
calculate_xP_and_confusion(model, model_variables_gdr1, model_name_gdr1, test_model_gdr1, models_eval)


# Goal distance reduction
test_model_gdr1a = smf.glm(formula="successful ~ goal_distance_end + angle_end" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_gdr1a.summary())

model_variables_gdr1a = ['goal_distance_end', 'angle_end']
model_name_gdr1a = 'test_model_gdr1a'
calculate_xP_and_confusion(model, model_variables_gdr1a, model_name_gdr1a, test_model_gdr1a, models_eval)


# ALL INCLUDED
test_model_allp = smf.glm(formula="successful ~ position_x_init_nm + position_y_init_nm + position_x_end_nm + position_y_end_nm + position_x_init + position_y_init + position_x_end + position_y_end + goal_distance + goal_distance_end + pass_distance + angle_init + angle_end + distance_x_init + distance_y_init + distance_x_end + distance_y_end + distance_x_init_sq + distance_y_init_sq + distance_x_end_sq + distance_y_end_sq + goal_distance_reduction" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_allp.summary())

model_variables_allp = ['position_x_init_nm', 'position_y_init_nm', 'position_x_end_nm', 'position_y_end_nm', 'position_x_init', 'position_y_init', 'position_x_end', 'position_y_end', 'goal_distance', 'goal_distance_end', 'pass_distance', 'angle_init', 'angle_end', 'distance_x_init', 'distance_y_init', 'distance_x_end', 'distance_y_end', 'distance_x_init_sq', 'distance_y_init_sq', 'distance_x_end_sq', 'distance_y_end_sq', 'goal_distance_reduction']
model_name_allp = 'test_model_allp'
calculate_xP_and_confusion(model, model_variables_allp, model_name_allp, test_model_allp, models_eval)

# Positions, distances and angles
test_model_allf = smf.glm(formula="successful ~ position_x_init + position_y_init + position_x_end + position_y_end + goal_distance + goal_distance_end + pass_distance + angle_init + angle_end + distance_x_init + distance_y_init + distance_x_end + distance_y_end + goal_distance_reduction + in_box" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_allf.summary())

model_variables_allf = ['position_x_init', 'position_y_init', 'position_x_end', 'position_y_end', 'goal_distance', 'goal_distance_end', 'pass_distance', 'angle_init', 'angle_end', 'distance_x_init', 'distance_y_init', 'distance_x_end', 'distance_y_end', 'goal_distance_reduction', 'in_box']
model_name_allf = 'test_model_allf'
calculate_xP_and_confusion(model, model_variables_allf, model_name_allf, test_model_allf, models_eval)


# Goal distance reduction
test_model_angles = smf.glm(formula="successful ~ goal_distance_end + angle_end + angle_init" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_angles.summary())

model_variables_angles = ['goal_distance_end', 'angle_end', 'angle_init']
model_name_angles = 'test_model_gdr1'
calculate_xP_and_confusion(model, model_variables_angles, model_name_angles, test_model_angles, models_eval)

# Box, angles and goal distance reduction
test_model_box = smf.glm(formula="successful ~ in_box + angle_end + angle_init + position_x_init + position_y_init + goal_distance_reduction" , data=model, 
                           family=sm.families.Binomial()).fit()
print(test_model_box.summary())

model_variables_box = ['in_box', 'angle_end', 'angle_init', 'position_x_init', 'position_y_init', 'goal_distance_reduction']
model_name_box = 'test_model_box'
calculate_xP_and_confusion(model, model_variables_box, model_name_box, test_model_box, models_eval)

# Including xGoals model
'''xGoals = pd.read_csv(wyscout_events+'/xGoals.csv')
xGoals = xGoals.rename(columns={"Y": "position_y_end_nm"})
result = pd.concat([model, xGoals], axis=1, join='inner')
# xGoal and starting position (x)
test_model_xG = smf.glm(formula="successful ~ xG + position_x_init" , data=result, 
                           family=sm.families.Binomial()).fit()
print(test_model_xG.summary())

model_variables_xG = ['xG', 'position_x_init']
model_name_xG = 'test_model_xG'
calculate_xP_and_confusion(result, model_variables_xG, model_name_xG, test_model_xG, models_eval)'''

# Calculating percentages and accuracy for each model
tp_per = np.asarray(models_eval['tp'])/np.asarray(models_eval['total'])
tn_per = np.asarray(models_eval['tn'])/np.asarray(models_eval['total'])
fp_per = np.asarray(models_eval['fp'])/np.asarray(models_eval['total'])
fn_per = np.asarray(models_eval['fn'])/np.asarray(models_eval['total'])
accuracy = (np.asarray(models_eval['tp']) + np.asarray(models_eval['tn']))/np.asarray(models_eval['total'])
false = fp_per + fn_per
models_eval['tp_per'] = tp_per
models_eval['tn_per'] = tn_per
models_eval['fp_per'] = fp_per
models_eval['fn_per'] = fn_per
models_eval['accuracy'] = accuracy
models_eval['false_per'] = false


models_eval

# Import players data and create evaluation dataframe
wyscout_players = 'Wyscout/'
players = pd.read_json(wyscout_players+'/players.json')
col_names = [['player_name', 'player_id', 'position', 'model_name', 'tp', 'tn', 'fp', 'fn', 'total', 'expected_success', 'real_success', 'expected_to_real']]
player_eval = pd.DataFrame(columns=col_names)

# Add calculated data to the players evaluation dataframe
def add_to_dict_pl(model_name, tp, tn, fp, fn, total, player_eval, player_name, player_id, position):
    player_eval.loc[-1] = [player_name, player_id, position, model_name, tp, tn, fp, fn, total, (tp+fp)/total, (tp+fn)/total, ((tp+fp) - (tp+fn))/total]
    player_eval.index = player_eval.index + 1
    player_eval = player_eval.sort_index() 

# Calculate confusion matrix
def confusion_matrix_pl(model, model_name, player_eval, player_name, player_id, position):
    tp = len(model[(model.successful == 1) & (model[model_name] > 0.5)])
    tn = len(model[(model.successful == 0) & (model[model_name] < 0.5)])
    fp = len(model[(model.successful == 0) & (model[model_name] > 0.5)])
    fn = len(model[(model.successful == 1) & (model[model_name] < 0.5)])
    total = tp + tn + fp + fn
    add_to_dict_pl(model_name, tp, tn, fp, fn, total, player_eval, player_name, player_id, position)

# Help function for calculating expected and actual passes and confusion matrix
def calculate_xP_and_confusion_pl(model, model_vars, model_name, test_model, player_eval, player_name, player_id, position):
    b=test_model.params
    xP=calculate_xP(model, model_vars,b)
    confusion_matrix_pl(model, model_name, player_eval, player_name, player_id, position)

# For each player, calculate the expected and actual percentage of correct passes
for pl in model['playerId'].unique():
    calculate_xP_and_confusion_pl(model[model['playerId']==pl], model_variables_xyab_end, model_name_xyab_end,                                  test_model_xyab_end, player_eval, players[players['wyId']==pl]['lastName'].tolist()[0],                                  pl, players[players['wyId']==pl].role.tolist()[0]['code3'])

player_eval[(np.asarray(player_eval['expected_to_real']) < 0.06) & (np.asarray(player_eval['total']) > 100) & (np.asarray(player_eval['position']) != 'DEF')]

pl_box = {}
for pl in player_eval['player_id'][(np.asarray(player_eval['expected_to_real']) < 0.06) & (np.asarray(player_eval['total']) > 100) & (np.asarray(player_eval['position']) != 'DEF')].values:
    pl_box[int(len(model[(model.playerId==pl[0]) & (model.in_box==1) & (model.successful==1)]))] = pl[0]


pl_box[max(list(pl_box.keys()))]
pl_box



sane_passes = model[model.playerId==245364]
pitchLengthX = 105
pitchWidthY = 65
lala = []
(fig,ax) = FCPython.createPitch(pitchLengthX,pitchWidthY,'yards','gray')
for i, cur in sane_passes.iterrows():
    if cur['successful'] == 1 and cur['in_box'] == 1:
        x=cur['position_x_init']
        y=cur['position_y_init']
        dx=cur['position_x_end'] - x
        dy=cur['position_y_end'] - y
        ax.arrow(x, pitchWidthY-y, dx, -dy, width = 0.4, color='red')
plt.text(5,75,'sane' + ' passes')
fig.set_size_inches(10, 7)
plt.show()

player_eval['expected_to_real'] = player_eval['expected_to_real'].astype(float)
player_eval['player_id'] = player_eval['player_id'].astype(int)



