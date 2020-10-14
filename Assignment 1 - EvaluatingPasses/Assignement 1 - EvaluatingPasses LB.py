# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:57:50 2020

@author: lodve
"""
from enum import IntEnum
from os import path, makedirs
from sklearn import linear_model
from sklearn.calibration import calibration_curve

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mplsoccer import pitch as mpl_pitch
from soccerplots.radar_chart import Radar

#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.metrics as skm
import matplotlib.pyplot as plt

ROOT_PATH_WYSCOUT = '.'
OUTPUT_PATH = 'assignment1-lodve_berre'
SAVE_OUTPUT = True
VIZ = True


### Enums for WyScout tags. ###
class WyScoutEvents(IntEnum):
    
    DUEL = 1
    FOUL = 2
    SETPIECE = 3
    GOALIE_OFF = 4
    INTERRUPTION = 5
    OFFSIDE = 6
    OTHER_ON_BALL = 7
    PASS = 8
    SAVE_ATTEMPT = 9
    SHOT = 10


class WyScoutSubEvents(IntEnum):
    
    AIR_DUEL = 10
    GROUND_ATT_DUEL = 11
    GROUND_DEF_DUEL = 12
    GROUND_LOO_DUEL = 13
    FOUL = 20
    HAND_FOUL = 21
    LATE_CARD_FOUL = 22
    OUT_OF_GAME_FOUL = 23
    PROTEST = 24
    SIMULATION = 25
    TIME_LOST = 26
    VIOLENT_FOUL = 27
    CORNER = 30
    FREE_KICK = 31
    FREE_KICK_CROSS = 32
    FREE_KICK_SHOT = 33
    GOAL_KICK = 34
    PENALTY = 35
    THROW_IN = 36
    GOALIE_OFF = 40
    BALL_OFF_FIELD = 50
    WHISTLE = 51
    OFFISIDE = 60
    ACCELERATION = 70
    CLEARANCE = 71
    TOUCH = 72
    CROSS = 80
    HAND_PASS = 81
    HEAD_PASS = 82
    HIGH_PASS = 83
    LAUNCH = 84
    SIMPLE_PASS = 85
    SMART_PASS = 86
    REFLEX = 90
    SAVE_ATTEMPT = 91
    SHOT = 100
    
    
    
class WyScoutTags(IntEnum):
    
    GOAL = 101
    OWN_GOAL = 102
    OPPORTUNITY = 201
    ASSIST = 301
    KEY_PASS = 302
    LEFT_FOOT = 401
    RIGHT_FOOT = 402
    HEAD_BODY = 403
    FREE_SPACE_R = 501
    FREE_SPACE_L = 502
    TAKE_ON_L = 503
    TAKE_ON_R = 504
    ANTICIPATED = 601
    ANTICIPATION = 602
    LOST = 701
    NEUTRAL = 702
    WON = 703
    HIGH = 801
    LOW = 802
    THROUGH = 901
    FAIRPLAY = 1001
    DIRECT = 1101
    INDIRECT = 1102
    GOAL_LOW_CENTER = 1201
    GOAL_LOW_RIGHT = 1202
    GOAL_CENTER = 1203
    GOAL_CENTER_LEFT = 1204
    GOAL_LOW_LEFT = 1205
    GOAL_CENTER_RIGHT = 1206
    GOAL_HIGH_CENTER = 1207
    GOAL_HIGH_LEFT = 1208
    GOAL_HIGH_RIGHT = 1209
    OUT_LOW_RIGHT = 1210
    OUT_CENTER_LEFT = 1211
    OUT_LOW_LEFT = 1212
    OUT_CENTER_RIGHT = 1213
    OUT_HIGH_CENTER = 1214
    OUT_HIGH_LEFT = 1215
    OUT_HIGH_RIGHT = 1216
    POST_LOW_RIGHT = 1217
    POST_CENTER_LEFT = 1218
    POST_LOW_LEFT = 1219
    POST_CENTER_RIGHT = 1220
    POST_HIGH_CENTER = 1221
    POST_HIGH_LEFT = 1222
    POST_HIGH_RIGHT = 1223
    FEINT = 1301
    MISSED_BALL = 1302
    INTERCEPTION = 1401
    CLEARANCE = 1501
    SLIDING_TACKLE = 1601
    RED_CARD = 1701
    SECOND_YELLOW_CARD = 1702
    YELLOW_CARD = 1703
    ACCURATE = 1801
    NOT_ACCURATE = 1802
    COUNTER_ATTAK = 1901
    DANGEROUS_BALL_LOST = 2001
    BLOCKED = 2101


##############################################################################
### --------------------- START WYSCOUT META METHODS --------------------- ###
##############################################################################
    
def expandWyScoutTags(dataFrame):
    
    tag_list = []
    
    for row in dataFrame.itertuples():
        
        tags = []
        
        for tag in row.tags:
            
            tags.append(list(tag.values())[0])
            
        tag_list.append(tags)

    return tag_list


def wyScoutEvents(country, eventTypes=None, teamIDs=None, matchIDs=None):
    """
    Details on WyScout tags are found here:
        https://apidocs.wyscout.com/matches-wyid-events
        
    WyScout x/y locations are in % from the left corner of the of the attacking team.
    """
    
    event_path = path.join(ROOT_PATH_WYSCOUT, 'events', 'events_%s.json' % country)
    events = pd.read_json(event_path)
    
    if not eventTypes is None:

        if type(eventTypes) is not list:

            eventTypes = [eventTypes]

        events = events[events.eventId.isin(eventTypes)]
        
    if not teamIDs is None:

        if type(teamIDs) is not list:

            teamIDs = [teamIDs]

        events = events[events.teamId.isin(teamIDs)]

    if not matchIDs is None:

        if type(matchIDs) is not list:

            matchIDs = [matchIDs]

        events = events[events.matchId.isin(matchIDs)]

    
    # Get locations on a familiar format.
    events['num_locations'] = events.positions.apply(lambda x: len(x))
    
    events['locations'] = events.positions.apply(lambda x: x[0])
    events['location_x'] = events.locations.apply(lambda x: x.get('x'))
    events['location_y'] = events.locations.apply(lambda x: x.get('y'))
    
    events['end_locations'] = events.apply(lambda x: x.positions[1] if x.num_locations == 2 else {}, axis=1)
    events['end_location_x'] = events.end_locations.apply(lambda x: x.get('x', None))
    events['end_location_y'] = events.end_locations.apply(lambda x: x.get('y', None))
    
    events.drop(columns=['num_locations', 'positions', 'locations', 'end_locations'], inplace=True)
    tags = expandWyScoutTags(events) # Merge list of dicts to one list
    events['tags'] = tags

    return events


def wyScoutMatches(country, matchIDs=None, teamIDs=None):
    
    matches_path = path.join(ROOT_PATH_WYSCOUT, 'matches', 'matches_%s.json' % country)
    matches = pd.read_json(matches_path)
    
    if matchIDs is not None:
        
        if type(matchIDs) is not list:
            
            matchIDs = list(matchIDs)
    
        matches = matches[matches.wyId.isin(matchIDs)]

    if teamIDs is not None:
        
        if type(teamIDs) is not list:
            
            teamIDs = list(teamIDs)
    
        matches = matches[matches.teamId.isin(teamIDs)]
    
    return matches


def wyScoutPlayers(playerIDs=None, teamIDs=None):
    
    players_path = path.join(ROOT_PATH_WYSCOUT, 'players.json')
    players = pd.read_json(players_path)
    players.shortName = players.shortName.str.decode('unicode_escape')
    
    if playerIDs is not None:
        
        if type(playerIDs) is not list:
            
            playerIDs = list(playerIDs)
    
        players = players[players.wyId.isin(playerIDs)]
    
    if teamIDs is not None:
        
        if type(teamIDs) is not list:
            
            teamIDs = list(teamIDs)
    
        players = players[players.currentTeamId.isin(teamIDs)]

    return players


def wyScoutTeams():
    
    teams_path = path.join(ROOT_PATH_WYSCOUT, 'teams.json')
    teams = pd.read_json(teams_path)
    
    teams.name = teams.name.str.decode('unicode_escape')
    
    return teams

##############################################################################
### ---------------------- END WYSCOUT META METHODS ---------------------- ###
##############################################################################


def get_xGModel(shots):
    """
    Train a naïve xG model on the shots.
    We'll use this later for looking at the dangerousity (is that a word?)
    of both the start and end location of the passes.
    """
    xG_model = linear_model.LogisticRegression(C=10000, random_state=0, multi_class='ovr', max_iter=10000)
    # Remove headers, we're not looking at those for the passing model either.
    headers = np.asarray([WyScoutTags.HEAD_BODY in tag for tag in shots.tags])
    
    shots = shots[np.invert(headers)]
    goals = np.asarray([WyScoutTags.GOAL in tag for tag in shots.tags])
    x = (100-shots.location_x.values)*1.04
    c = (shots.location_y.values-50)*.69
    dist = (x**2+c**2)**0.5
    theta = np.arctan(7.32*x/(x**2 + c**2-(7.32/2)**2))
    theta[theta < 0] += np.pi
    
    shots['goal_angle'] = theta
    shots['goal_dist'] = dist
    shots['accurate'] = goals

    xG_model.fit(shots[['goal_angle', 'goal_dist']], shots.accurate)

    return shots, xG_model
    

def passProbPerVar(passes, columns, ranges, title=""):
    """
    Look at each given variable in columns and see how the correlation
    between the variable and pass success is.
    """
    
    fig, axes = plt.subplots(4, 4, sharey=True, figsize=(11.69, 8.27), dpi=150)
    cf_fig, cf_axes = plt.subplots(4, 4, figsize=(11.69, 8.27), dpi=150)
    # Setting up sklearn logistic regression to match statsmodels logistic regression.
    model = linear_model.LogisticRegression(C=10000, random_state=0, multi_class='ovr')
    train_sample = passes.sample(10000)
    test_sample = passes.sample(10000)
    class_names=[0, 1] # name  of classes
    tick_marks = np.arange(len(class_names))+0.5
    
      
    for col, ran, ax, cf_ax in zip(columns, ranges, fig.axes, cf_fig.axes):
        
        ind_var = train_sample[col].values
        ind_var_suc = train_sample[train_sample.accurate == 1][col].values
        ind_var_fail = train_sample[train_sample.accurate == 0][col].values
        
        pass_dist = np.histogram(ind_var, bins=50, range=ran)
        pass_suc_dist = np.histogram(ind_var_suc, bins=50, range=ran)
        # pass_suc_prob = np.divide(pass_suc_dist[0], pass_dist[0], out=np.zeros_like(pass_suc_dist[0]), where=pass_dist[0]!=0)
        pass_suc_prob = np.divide(pass_suc_dist[0], pass_dist[0])
        mid_range = (pass_dist[1][:-1] + pass_dist[1][1:])/2
        ax.scatter(mid_range, pass_suc_prob, marker='.', s=10, color='black', zorder=3)
        ax.scatter(ind_var_suc, train_sample[train_sample.accurate==1].accurate.values, marker='.', s=10, color='blue', zorder=2)
        ax.scatter(ind_var_fail, train_sample[train_sample.accurate==0].accurate.values, marker='.', s=10, color='red', zorder=1)
        ax.set_ylabel('P(PS)')
        ax.set_xlabel(col)
        ax.set_xlim(ran[0], ran[1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(linewidth=0.25, zorder=1)

        model.fit(ind_var.reshape(-1, 1), train_sample.accurate.values)
        x0 = model.intercept_[0]
        x1 = model.coef_[0][0]
        
        step = (ran[1]-ran[0])/len(ind_var)
        x = np.arange(start=ran[0], stop=ran[1], step=step)
        
        # For some reason, some dimensions don't always match. #workaround
        if len(ind_var) > len(x):
            
            x = np.arange(start=ran[0], stop=ran[1]+step, step=step)
            
        elif len(ind_var) < len(x):
            
            x = np.arange(start=ran[0], stop=ran[1]-step, step=step)
            
        y = 1/(1+np.exp(x0+x1*x))
        
        ax.plot(x, y, linewidth=3, color='red', alpha=0.3, zorder=1)
        ax.plot(x, 1-y, linewidth=3, color='blue', alpha=0.3, zorder=2)

        # Plot the confusion matrix.
        pass_pred = model.predict(test_sample[col].values.reshape(-1,1))
        cnf_matrix = skm.confusion_matrix(test_sample.accurate, pass_pred)/100
        ac_s = skm.accuracy_score(test_sample.accurate, pass_pred)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cbar=False, fmt='g', ax=cf_ax)
        cf_ax.set_xticks(tick_marks)
        cf_ax.set_yticks(tick_marks)
        cf_ax.set_xticklabels(class_names)
        cf_ax.set_yticklabels(class_names)
        cf_ax.set_title('Accuracy score: %.2f' % (ac_s*100), y=1.1)
        cf_ax.set_xlabel(col)
    
    fig.suptitle(title)
    fig.tight_layout()
    cf_fig.suptitle(title)
    cf_fig.tight_layout()
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, 'SingleVariablePredictionStrength-'+title+'.jpg'))
        cf_fig.savefig(path.join(OUTPUT_PATH, 'SingleVariableConfusionMatrix-'+title+'.jpg'))


def testModels(passes, models, title):
    """
    Test various models and plot their outcome against each other.
    """
    
    cm_fig, cm_axes = plt.subplots(4, 4, figsize=(11.69, 8.27), dpi=150)
    cc_fig, cc_axes = plt.subplots(4, 4, figsize=(11.69, 8.27), dpi=150)
    sk_model = linear_model.LogisticRegression(C=10000, random_state=0, multi_class='ovr', max_iter=10000)
    train_sample = passes.sample(10000, random_state=0)
    test_sample = passes.sample(10000, random_state=0)
    class_names=[0, 1] # name  of classes
    tick_marks = np.arange(len(class_names))+0.5
    
    for model, cm_ax, cc_ax in zip(models, cm_fig.axes, cc_fig.axes):
        
        m_ind = (models.index(model)+1)
        
        m_str = "Now testing model %d:" % m_ind
        m_str += "\n" + "%s" % "-"*80
        m_str += "\nParameters used:"
        m_str += "\n" + "+".join(model)
        m_str += "\n" + "%s" % "-"*80
        
        sm_model = smf.glm(formula='accurate ~ '+ ' + '.join(model),
                           data=train_sample,
                           family=sm.families.Binomial()).fit()
        m_str += "\n" + str(sm_model.summary())
        ind_var = train_sample[model]
        sk_model.fit(ind_var, train_sample.accurate.values)
        
        
        # Plot the confusion matrix.
        pass_pred = sk_model.predict(test_sample[model])
        cnf_matrix = skm.confusion_matrix(test_sample.accurate, pass_pred)/100
        ac_s = skm.accuracy_score(test_sample.accurate, pass_pred)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cbar=False, fmt='g', ax=cm_ax, cmap='RdBu')
        cm_ax.set_xticks(tick_marks)
        cm_ax.set_yticks(tick_marks)
        cm_ax.set_xticklabels(class_names)
        cm_ax.set_yticklabels(class_names)
        cm_ax.set_title('Accuracy score: %.2f' % (ac_s*100), y=1.1)
        cm_ax.set_xlabel('Model %s' % m_ind)
        
        # Plot the calibration curve.
        cc_ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        pass_pred_prob = sk_model.predict_proba(test_sample[model].values)[:, 1]
        clf_score = skm.brier_score_loss(test_sample.accurate.values, pass_pred_prob, pos_label=test_sample.accurate.values.max())
        pre_score = skm.precision_score(test_sample.accurate.values, pass_pred)
        rec_score = skm.recall_score(test_sample.accurate.values, pass_pred)
        fon_score = skm.f1_score(test_sample.accurate.values, pass_pred)
       
        m_str += "\nBrier score: %1.3f" % clf_score
        m_str += "\nPrecision score: %1.3f" % pre_score
        m_str += "\nRecall score: %1.3f" % rec_score
        m_str += "\nF1 score: %1.3f" % fon_score
        
        frac_pos, mean_pred_val = calibration_curve(test_sample.accurate, pass_pred_prob, n_bins=10)
        cc_ax.plot(mean_pred_val, frac_pos, 's-', label='%1.3d' % clf_score)
        cc_ax.set_ylabel('Fraction of positives')
        cc_ax.set_ylim([-0.05, 1.05])
        cc_ax.set_title('Calibration curve M%d' % m_ind)
        m_str += "\n" + "%s" % "-"*80
        
        if VIZ:
            
            print(m_str)
            
        if SAVE_OUTPUT:
            
            with open(path.join(OUTPUT_PATH, 'ModelTestSummary_M%d.txt' % m_ind), 'w') as mtest_file:
                
                mtest_file.write(m_str)

    cm_fig.suptitle(title)
    cm_fig.tight_layout()
    cc_fig.suptitle(title)
    cc_fig.tight_layout()
    
    if SAVE_OUTPUT:
        
        cm_fig.savefig(path.join(OUTPUT_PATH, "ModelTest_ConfusionMatrix" + title + ".png"))
        cc_fig.savefig(path.join(OUTPUT_PATH, "ModelTest_CalibrationCurve" + title + ".png"))


def testModelsForReport(passes, models, titles):
    """
    Test various models and plot their outcome against each other.
    """
    
    fig, axes = plt.subplots(2, 4, figsize=(12,6), dpi=300)
    sk_model = linear_model.LogisticRegression(C=10000, random_state=0, multi_class='ovr', max_iter=10000)
    train_sample = passes.sample(10000, random_state=0)
    test_sample = passes.sample(10000, random_state=0)
    class_names=[0, 1] # name  of classes
    tick_marks = np.arange(len(class_names))+0.5
    
    for index, model in enumerate(models):

        cm_ax = axes[0, index]
        cc_ax = axes[1, index]
        
        ind_var = train_sample[model]
        sk_model.fit(ind_var, train_sample.accurate.values)
        
        # Plot the confusion matrix.
        pass_pred = sk_model.predict(test_sample[model])
        cnf_matrix = skm.confusion_matrix(test_sample.accurate, pass_pred)/100
        ac_s = skm.accuracy_score(test_sample.accurate, pass_pred)
        sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cbar=False, fmt='g', ax=cm_ax, cmap='RdBu')
        cm_ax.set_xticks(tick_marks)
        cm_ax.set_yticks(tick_marks)
        cm_ax.set_xticklabels(class_names)
        cm_ax.set_yticklabels(class_names)
        cm_ax.set_title('Accuracy score %s: %.2f' % (titles[index], ac_s*100))
        cm_ax.set_aspect('equal', 'box')
        
        # Plot the calibration curve.
        cc_ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated', zorder=2)
        pass_pred_prob = sk_model.predict_proba(test_sample[model].values)[:, 1]
        clf_score = skm.brier_score_loss(test_sample.accurate.values, pass_pred_prob, pos_label=test_sample.accurate.values.max())
       
        frac_pos, mean_pred_val = calibration_curve(test_sample.accurate, pass_pred_prob, n_bins=10)
        cc_ax.plot(mean_pred_val, frac_pos, 's-', label='%1.3d' % clf_score, zorder=3)
        cc_ax.set_ylabel('Fraction of positives')
        cc_ax.set_ylim([-0.05, 1.05])
        cc_ax.set_title('Calibration curve %s' % titles[index])
        cc_ax.grid(linewidth=0.25, zorder=1)
        cc_ax.set_aspect('equal', 'box')
        
    fig.tight_layout()
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, "ModelTest_Comparisons.png"), bbox_inches='tight', dpi=300)


def evaluateVariables(passes):
    """
    Used to evaluate the predictive strength of each variables.
    Most to the plots here are not included in the report, but were key to
    the decision making regarding which variables to include in the final
    model.
    """
    columns_of_interest = ['location_x',
                           'location_y',
                           'location_x_sq',
                           'xy',
                           'end_location_x',
                           'end_location_y',
                           'end_xy',
                           'length',
                           'angle',
                           'goal_dist',
                           'goal_angle',
                           'end_goal_angle',
                           'delta_goal_angle',
                           'xG',
                           'end_xG',
                           'xG_added']
    
    ranges = [(0,100),
              (0,100),
              (0,10000),
              (0,10000),
              (0,100),
              (0,100),
              (0,10000),
              (0,125),
              (-np.pi/2,np.pi/2),
              (0,110),
              (0,np.pi/2),
              (0,np.pi/2),
              (-np.pi/2,np.pi/2),
              (0,1),
              (0,1),
              (-0.5,0.5)]


    # See if we can find prediction strength for single variables.
    passProbPerVar(passes, columns_of_interest, ranges, "All passes")
    # Test for non-simple passes only.
    passProbPerVar(passes[passes.simple == 0], columns_of_interest, ranges, "All passes excluding SIMPLE")
    # Test for passes ending in opposition half only.
    passProbPerVar(passes[passes.end_location_x > 50], columns_of_interest, ranges, "Passes ending in opposition half only")
    
    # Set up some models to test.
    m1 = ['location_x', 'location_y']
    m2 = ['end_location_x', 'end_location_y']
    m3 = m1 + m2
    m4 = ['xy', 'end_xy']
    m5 = ['location_x', 'end_location_x']
    m6 = m2 + ['length']
    m7 = m6 + ['xG_added']
    m8 = m6 + ['cross']
    m9 = m6 + ['angle']
    m10 = m9 + ['smart']
    m11 = m10 + ['delta_goal_angle']
    m12 = m11 + ['simple']
    m13 = m12 + ['xG']
    m14 = m13 + ['end_xG']
    m15 = m14 + ['end_goal_angle']
    m16 = m15 + ['high']
    models = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16]
    
    # Test the models - first on the opposition half only, then all passes.
    testModels(passes[passes.end_location_x > 50], models, "Passes into opposition half only")
    testModels(passes, models, "All passes")
    testModelsForReport(passes[passes.end_location_x > 50], [m1, m3, m7, m14], ['Model 1', 'Model 3', 'Model 7', 'Model 14'])
    
    # From our evaluation, where the criterion is good prediction of difficult
    # passes, we conclude that m10 is the best model.
    return m1, m7

def plotModelVSModel(passes):
    
    good_passes = passes[passes.accurate == 1]
    good_passes_im = passes[passes.imPS == 1]
    good_passes_fm = passes[passes.mPS == 1]
    
    # Actual pass completion probability by start/end location from data.
    h_sl_all_passes, h_sl_all_xedges, h_sl_all_yedges = np.histogram2d(passes.location_x, passes.location_y, bins=50)
    h_sl_good_passes, h_sl_good_xedges, h_sl_good_yedges = np.histogram2d(good_passes.location_x, good_passes.location_y, bins=50)
    sl_good_ratio = np.divide(h_sl_good_passes, h_sl_all_passes, out=np.zeros_like(h_sl_good_passes), where=h_sl_all_passes!=0)
    h_el_all_passes, h_el_all_xedges, h_el_all_yedges = np.histogram2d(passes.end_location_x, passes.end_location_y, bins=50)
    h_el_good_passes, h_el_good_xedges, h_el_good_yedges = np.histogram2d(good_passes.end_location_x, good_passes.end_location_y, bins=50)
    el_good_ratio = np.divide(h_el_good_passes, h_el_all_passes, out=np.zeros_like(h_el_good_passes), where=h_el_all_passes!=0)

    # Pass completion probability by start/end location from initial model.
    h_sl_good_passes_im, h_sl_good_xedges_im, h_sl_good_yedges_im = np.histogram2d(good_passes_im.location_x, good_passes_im.location_y, bins=50)
    sl_good_ratio_im = np.divide(h_sl_good_passes_im, h_sl_all_passes, out=np.zeros_like(h_sl_good_passes_im), where=h_sl_all_passes!=0)
    h_el_good_passes_im, h_el_good_xedges_im, h_el_good_yedges_im = np.histogram2d(good_passes_im.end_location_x, good_passes_im.end_location_y, bins=50)
    el_good_ratio_im = np.divide(h_el_good_passes_im, h_el_all_passes, out=np.zeros_like(h_el_good_passes_im), where=h_el_all_passes!=0)

    # Pass completion probability by start/end location from final model.
    h_sl_good_passes_fm, h_sl_good_xedges_fm, h_sl_good_yedges_fm = np.histogram2d(good_passes_fm.location_x, good_passes_fm.location_y, bins=50)
    sl_good_ratio_fm = np.divide(h_sl_good_passes_im, h_sl_all_passes, out=np.zeros_like(h_sl_good_passes_fm), where=h_sl_all_passes!=0)
    h_el_good_passes_fm, h_el_good_xedges_im, h_el_good_yedges_fm = np.histogram2d(good_passes_fm.end_location_x, good_passes_fm.end_location_y, bins=50)
    el_good_ratio_fm = np.divide(h_el_good_passes_im, h_el_all_passes, out=np.zeros_like(h_el_good_passes_fm), where=h_el_all_passes!=0)

    # This is the one for the report.
    fig, ax = plt.subplots(3, 2, figsize=(11.69, 8.27), dpi=300)
    d_sl_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    d_el_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    im_sl_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    im_el_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    fm_sl_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    fm_el_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)

    d_sl_pitch.draw(ax=ax[0][0])
    d_el_pitch.draw(ax=ax[0][1])
    im_sl_pitch.draw(ax=ax[1][0])
    im_el_pitch.draw(ax=ax[1][1])
    fm_sl_pitch.draw(ax=ax[2][0])
    fm_el_pitch.draw(ax=ax[2][1])

    pcm = ax[0][0].imshow(sl_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax[0][0].set_title('Actual P(PS) vs Start Location', size=15)
    ax[0][0].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[0][0], fraction=0.05, orientation='vertical')
    pcm = ax[0][1].imshow(el_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax[0][1].set_title('Actual P(PS) vs End Location', size=15)
    ax[0][1].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[0][1], fraction=0.05, orientation='vertical')
   
    pcm = ax[1][0].imshow(sl_good_ratio_im.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r', vmin=0, vmax=1)
    ax[1][0].set_title('Intial Model P(PS) vs Start Location', size=15)
    ax[1][0].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[1][0], fraction=0.05, orientation='vertical')
    pcm = ax[1][1].imshow(el_good_ratio_im.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r', vmin=0, vmax=1)
    ax[1][1].set_title('Initial Model P(PS) vs End Location', size=15)
    ax[1][1].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[1][1], fraction=0.05, orientation='vertical')
    
    pcm = ax[2][0].imshow(sl_good_ratio_fm.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r', vmin=0, vmax=1)
    ax[2][0].set_title('Final Model P(PS) vs Start Location', size=15)
    ax[2][0].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[2][0], fraction=0.05, orientation='vertical')
    pcm = ax[2][1].imshow(el_good_ratio_fm.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r', vmin=0, vmax=1)
    ax[2][1].set_title('Final Model P(PS) vs End Location', size=15)
    ax[2][1].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[2][1], fraction=0.05, orientation='vertical')
    
    fig.tight_layout()
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, "ActualData_vs_IntialModel_vs_FinalModel.png"), dpi=300)



def plotPassHeatMaps(passes, shots, title):
    
    good_passes = passes[passes.accurate == 1]
    bad_passes = passes[passes.accurate == 0]

    # Goal probability vs location.    
    good_shots = shots[shots.accurate == 1]
    h_shots, h_shots_xedges, h_shots_yedges = np.histogram2d(shots.location_x, shots.location_y, bins=50)
    h_good_shots, h_good_shots_xedges, h_good_shots_yedges = np.histogram2d(good_shots.location_x, good_shots.location_y, bins=50)
    good_shots_ratio = np.divide(h_good_shots, h_shots, out=np.zeros_like(h_good_shots), where=h_shots!=0)
    
    # Look at pass completion rate by start location zone.
    h_sl_all_passes, h_sl_all_xedges, h_sl_all_yedges = np.histogram2d(passes.location_x, passes.location_y, bins=50)
    h_sl_good_passes, h_sl_good_xedges, h_sl_good_yedges = np.histogram2d(good_passes.location_x, good_passes.location_y, bins=50)
    h_sl_bad_passes, h_sl_bad_xedges, h_sl_bad_yedges = np.histogram2d(bad_passes.location_x, bad_passes.location_y, bins=50)

    sl_good_ratio = np.divide(h_sl_good_passes, h_sl_all_passes, out=np.zeros_like(h_sl_good_passes), where=h_sl_all_passes!=0)
    sl_bad_ratio = np.divide(h_sl_bad_passes, h_sl_all_passes, out=np.zeros_like(h_sl_bad_passes), where=h_sl_all_passes!=0)

    # Look at pass completion rate by end location zone.
    h_el_all_passes, h_el_all_xedges, h_el_all_yedges = np.histogram2d(passes.end_location_x, passes.end_location_y, bins=50)
    h_el_good_passes, h_el_good_xedges, h_el_good_yedges = np.histogram2d(good_passes.end_location_x, good_passes.end_location_y, bins=50)
    h_el_bad_passes, h_el_bad_xedges, h_el_bad_yedges = np.histogram2d(bad_passes.end_location_x, bad_passes.end_location_y, bins=50)

    el_good_ratio = np.divide(h_el_good_passes, h_el_all_passes, out=np.zeros_like(h_el_good_passes), where=h_el_all_passes!=0)
    el_bad_ratio = np.divide(h_el_bad_passes, h_el_all_passes, out=np.zeros_like(h_el_bad_passes), where=h_el_all_passes!=0)

    # See where most passes are made.
    p_max = h_sl_all_passes.mean() + h_sl_all_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_sl_all_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass start locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassStartLocHeatMap.png"))
    
    # See where most passes succeed.
    p_max = h_sl_good_passes.mean() + h_sl_good_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_sl_good_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass success start locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassStartSucHeatMap.png"))
    
    # See where most passes fail.
    p_max = h_sl_bad_passes.mean() + h_sl_bad_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_sl_bad_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass failure start locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)    

    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassStartFailHeatMap.png"))

    # Pass success ratio.
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(sl_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax.set_title('P(Pass Success) vs start location', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('P(Pass Success)', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassStartSucRatioHeatMap.png"))

    # Pass failure ratio.
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(sl_bad_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax.set_title('P(Pass Failure) vs start location', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('P(Pass Failure)', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)

    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassStartFailRatioHeatMap.png"))

    # See where most passes end.
    p_max = h_el_all_passes.mean() + h_el_all_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_el_all_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass end locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassEndLocHeatMap.png"))
    
    # See where most passes succeed.
    p_max = h_el_good_passes.mean() + h_el_good_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_el_good_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass success end locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassEndSucHeatMap.png"))
    
    # See where most passes fail.
    p_max = h_el_bad_passes.mean() + h_el_bad_passes.std()*2
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(h_el_bad_passes.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='hot', vmin=0, vmax=p_max)
    ax.set_title('Pass failure end locations', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('# of passes', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)    

    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassEndFailHeatMap.png"))

    # Pass success ratio.
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(el_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax.set_title('P(Pass Success) vs end location', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('P(Pass Success)', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassEndSucRatioHeatMap.png"))

    # Pass failure ratio.
    pitch = mpl_pitch.Pitch(figsize=(16,11),
                            pitch_type='wyscout',
                            orientation='horizontal',
                            pitch_color='#19232D',
                            line_color='#19232D',
                            view='full',
                            tight_layout=True,
                            line_zorder=2)
    fig, ax = pitch.draw()
    fig.set_facecolor('#19232D')
    pcm = ax.imshow(el_bad_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax.set_title('P(Pass Failure) vs end location', color='lightgrey', size=30)
    cb = fig.colorbar(pcm, ax=ax, orientation='vertical')
    cb.set_label('P(Pass Failure)', color='lightgrey', size=20)
    cb.ax.yaxis.set_tick_params(color='lightgrey')
    cb.outline.set_edgecolor('lightgrey')
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='lightgrey', size=20)
    ax.set_aspect(69/104)

    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - PassEndFailRatioHeatMap.png"))

    # This is the one for the report.
    fig, ax = plt.subplots(1, 3, figsize=(11.69, 8.27), dpi=300)
    gs_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    sl_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)
    el_pitch = mpl_pitch.Pitch(pitch_type='wyscout', view='half', line_zorder=2)

    gs_pitch.draw(ax=ax[0])
    sl_pitch.draw(ax=ax[1])
    el_pitch.draw(ax=ax[2])

    pcm = ax[0].imshow(good_shots_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r', vmin=0, vmax=0.5)
    ax[0].set_title('P(Goal) vs Shot Location', size=15)
    ax[0].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[0], fraction=0.05, orientation='vertical')
    pcm = ax[1].imshow(sl_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax[1].set_title('P(PS) vs Start Location', size=15)
    ax[1].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[1], fraction=0.05, orientation='vertical')
    pcm = ax[2].imshow(el_good_ratio.T, origin='lower', extent=[0,100,0,100], zorder=1, cmap='RdBu_r')
    ax[2].set_title('P(PS) vs End Location', size=15)
    ax[2].set_aspect(69/104)
    cb = fig.colorbar(pcm, ax=ax[2], fraction=0.05, orientation='vertical')
    fig.tight_layout()
    
    if SAVE_OUTPUT:
        
        fig.savefig(path.join(OUTPUT_PATH, title + " - Goal_vs_PSStart_vs_PSEnd_HeatMap.png"), dpi=300)


def preProcessPasses(passes, xg_model):
    """
    The raw data set from WyScout doesn't have all the variables we want to
    look at for our passing model. This method adds all the derived variables
    we want to evaluate and possibly include in the final passing model.
    """
    # The accuracy is our dependent variable.
    # No column for this in WyScout data, must use list expansion on the list of tags.
    accurate_passes = np.asarray([WyScoutTags.ACCURATE in tag for tag in passes.tags]).astype(int)
    passes['accurate'] = accurate_passes
        
    # Extract good and bad passes only for baseline visualizations.
    good_passes = passes[passes.accurate == 1]
    bad_passes = passes[passes.accurate == 0]
    
    # Just as a baseline, let's see what the overall accuarcy in the dataset is.
    # Personal note: It seems a bit high, FBREF has the overall accuracy for
    # this particular PL season as  75.42%.
    print("Data set average pass accuracy: %.2f %%" % (len(good_passes)/(len(good_passes)+len(bad_passes))*100))
    

    passes['location_x_sq'] = passes.location_x**2
    passes['location_y_sq'] = passes.location_y**2
    
    # WyScout uses percentages of pitches as coordinates, so we must scale them
    # to obtain proper derived properties like distance and angles.
    # Average PL pitch size is 104x69 m.
    passes.location_x *= 1.04
    passes.location_y *= 0.69
    passes.end_location_x *= 1.04
    passes.end_location_y *= 0.69
    # Create derived variables.
    passes['xy'] = passes.location_x*passes.location_y
    passes['end_xy'] = passes.end_location_x*passes.end_location_y
    passes['length'] = ((passes.end_location_x - passes.location_x)**2+(passes.end_location_y-passes.location_y)**2)**0.5
    passes['angle'] = np.arctan2(passes.end_location_x - passes.location_x, np.abs(passes.end_location_y-passes.location_y))
    # Minor hack. Pretend end goal dist+angle is start due to xG model trained on those column names.
    # Fix actual goal dist and angle later.
    passes['goal_dist'] = ((104 - passes.end_location_x)**2+(34.5-passes.end_location_y)**2)**0.5
    passes['goal_angle'] = np.arctan((7.32*(104-passes.end_location_x))/(((104-passes.end_location_x)**2+(34.5-passes.end_location_y)**2)-(7.32/2)**2))
    passes['end_xG'] = xg_model.predict_proba(passes[['goal_angle', 'goal_dist']])[:, 1]
    passes['end_goal_angle'] = passes['goal_angle'].copy()
    passes['end_goal_dist'] = passes['goal_dist'].copy()
    passes['goal_dist'] = ((104 - passes.location_x)**2+(34.5-passes.location_y)**2)**0.5
    passes['goal_dist_sq'] = passes['goal_dist']**2
    passes['goal_angle'] = np.arctan((7.32*(104-passes.location_x))/(((104-passes.location_x)**2+(34.5-passes.location_y)**2)-(7.32/2)**2))
    passes['delta_goal_angle'] = passes['end_goal_angle'] - passes['goal_angle']
    passes['xG'] = xg_model.predict_proba(passes[['goal_angle', 'goal_dist']])[:, 1]
    passes['xG_added'] = passes['end_xG'] - passes['xG']

    # Expand classification tags.
    passes['simple'] = passes['subEventId'] == WyScoutSubEvents.SIMPLE_PASS
    passes['high'] = passes['subEventId'] == WyScoutSubEvents.HIGH_PASS
    passes['cross'] = passes['subEventId'] == WyScoutSubEvents.CROSS
    passes['smart'] = passes['subEventId'] == WyScoutSubEvents.SMART_PASS
    passes['launch'] = passes['subEventId'] == WyScoutSubEvents.LAUNCH
    
    # Scale back for plotting with mplsoccer.
    passes.location_x /= 1.04
    passes.location_y /= 0.69
    passes.end_location_x /= 1.04
    passes.end_location_y /= 0.69
    
    return passes
    

def investigateData(passes, match_data, player_data, team_data):
    """
    Fetch data per 90 and total numbers for all players.
    
    For this exercise, we've defined high risk passes (hr_passes) as passes with
    a probability of success < 25% and a high xG added pass as any pass with
    an xGA > 0.05.
    """
    
    def getMinutesPlayed(matches, match_ids, player_id, team_ids):
        """
        Internal helper to get minutes played for a player.
        This isn't entirely accurate, as it doesn't take into account how long
        the match actually lasted, and subs on overtime get credited for the
        amount of overtime. Should be close enough and not a major source of
        uncertainty for the purpose of this exercise.
        """
        minutes_played = 0
        
        for match_id in match_ids:
            
            teams_data = matches.loc[matches.wyId == match_id].teamsData.values[0]
            
            for team_id in team_ids:
                
                team_data = teams_data.get(str(team_id))
            
                if team_data is None:
                
                    continue
                
                else:
                    team_data = team_data.get('formation')
                    lineup = pd.json_normalize(team_data.get('lineup'))
                    subs = team_data.get('substitutions')
                    
                    if subs != 'null':
                        
                        subs = pd.json_normalize(subs)
                    
                        if player_id in subs.playerOut.values:
                        
                            try:
                                sub_off = subs[subs.playerOut == player_id].minute.values[0]
                                minutes_played += sub_off
                    
                            except:
                                print(match_id, team_id, player_id)
                                print(subs, player_id)
                    
                        elif player_id in subs.playerIn.values:
                        
                            try:
                                sub_in = subs[subs.playerIn == player_id].minute.values[0]
                                minutes_played += abs(90-sub_in)
                        
                            except:
                                
                                print(subs, player_id)
                                
                        elif player_id in lineup.playerId.values:
                        
                            minutes_played += 90
                            
                    else:
                        
                        minutes_played += 90
            
        return minutes_played

    
    def getPlayerMeta(players, playerId, teams, team_ids):
        """
        Convenience method for getting player short name and position.
        """
        
        player_data = players[players.wyId == playerId]
        name = player_data.shortName.values[0]
        team = []
        
        for team_id in team_ids:
        
            if team_id is None:

                team.append("Free agent")
        
            else:
            
                team.append(teams[teams.wyId == team_id].name.values[0])

        team = "/".join(team)
        position = pd.json_normalize(player_data.role).name.values[0]
        
        return name, team, position
    
    hr_pass_threshold = 0.2
    h_xGA_threshold = 0.1
    
    # Set up a dataframe for all players, to be used for collating stats.
    players = pd.DataFrame(passes.playerId.unique(), columns=['playerId'])
    
    columns = ['name', 'team', 'position', 'matches_played', 'minutes_played',
               'passes', 'pass_suc', 'xP_suc', 'xGA',
               'passes_p90', 'pass_suc_p90', 'xP_suc_p90', 'xGA_p90',
               'passes_F3', 'pass_suc_F3', 'xP_suc_F3', 'xGA_F3',
               'passes_F3_p90', 'pass_suc_F3_p90', 'xP_suc_F3_p90', 'xGA_F3_p90',
               'net_progression', 'net_progression_p90',
               'hr_passes', 'hr_pass_suc', 'hr_xP_suc', 'hr_xGA',
               'hr_passes_p90', 'hr_pass_suc_p90', 'hr_xP_suc_p90', 'hr_xGA_p90',
               'hxGA_passes', 'hxGA_pass_suc', 'hxGA_xP_suc','hxGA_xGA',
               'hxGA_passes_p90', 'hxGA_pass_suc_p90', 'hxGA_xP_suc_p90', 'hxGA_xGA_p90']
    empty_data = np.zeros((len(players), len(columns)))
    players[columns] = empty_data
    
    # Iterate through all players and collate data.
    for player in players.playerId.to_list():
        
        # For some reason I've got a player with ID 0 in here, don't have time
        # to investigate why, hence just working around it...
        if player == 0:
            
            continue
        
        pp = passes[passes.playerId == player]
        # Take mid-season transfers into account...
        team_ids = list(pp.teamId.unique())
        # Fetch a list of all matches in order to get P/90 stats.
        match_ids = list(pp.matchId.unique())
        minutes_played = getMinutesPlayed(match_data, match_ids, player, team_ids)
        name, team, position = getPlayerMeta(player_data, player, team_data, team_ids)
        players.loc[players.playerId == player, 'name'] = name
        players.loc[players.playerId == player, 'team'] = team
        
        if minutes_played == 0:
            
            print("Warning: No minutes found for player: " + name)            
            continue
        
        nineties_played = minutes_played/90
        players.loc[players.playerId == player, 'position'] = position
        players.loc[players.playerId == player, 'matches_played'] = len(pp.matchId.unique())
        players.loc[players.playerId == player, 'minutes_played'] = minutes_played
        
        players.loc[players.playerId == player, 'passes'] = len(pp)
        players.loc[players.playerId == player, 'pass_suc'] = pp.accurate.sum()/len(pp)
        players.loc[players.playerId == player, 'xP_suc'] = pp.xPS.sum()/len(pp)
        players.loc[players.playerId == player, 'xGA'] = pp.xG_added.sum()
        
        players.loc[players.playerId == player, 'passes_p90'] = len(pp)/nineties_played
        players.loc[players.playerId == player, 'pass_suc_p90'] = pp.accurate.sum()/nineties_played
        players.loc[players.playerId == player, 'xP_suc_p90'] = pp.xPS.sum()/nineties_played
        players.loc[players.playerId == player, 'xGA_p90'] = pp.xG_added.sum()/nineties_played
        
        passes_F3 = pp[pp.end_location_x > 66]
        
        if len(passes_F3) > 0:
            players.loc[players.playerId == player, 'passes_F3'] = len(passes_F3)
            players.loc[players.playerId == player, 'pass_suc_F3'] = passes_F3.accurate.sum()/len(passes_F3)
            players.loc[players.playerId == player, 'xP_suc_F3'] = passes_F3.xPS.sum()/len(passes_F3)
            players.loc[players.playerId == player, 'xGA_F3'] = passes_F3.xG_added.sum()
        
            players.loc[players.playerId == player, 'passes_F3_p90'] = len(passes_F3)/nineties_played
            players.loc[players.playerId == player, 'pass_suc_F3_p90'] = passes_F3.accurate.sum()/nineties_played
            players.loc[players.playerId == player, 'xP_suc_F3_p90'] = passes_F3.xPS.sum()/nineties_played
            players.loc[players.playerId == player, 'xGA_F3_p90'] = passes_F3.xG_added.sum()/nineties_played

        players.loc[players.playerId == player, 'net_progression'] = (pp.end_location_x-pp.location_x).sum()
        players.loc[players.playerId == player, 'net_progression_p90'] = (pp.end_location_x-pp.location_x).sum()/nineties_played

        hr_passes = pp[pp.xPS < hr_pass_threshold]
        
        if len(hr_passes) > 0:
            players.loc[players.playerId == player, 'hr_passes'] = len(hr_passes)
            players.loc[players.playerId == player, 'hr_pass_suc'] = hr_passes.accurate.sum()/len(passes_F3)
            players.loc[players.playerId == player, 'hr_xP_suc'] = hr_passes.xPS.sum()/len(passes_F3)
            players.loc[players.playerId == player, 'hr_xGA'] = hr_passes.xG_added.sum()
    
            players.loc[players.playerId == player, 'hr_passes_p90'] = len(hr_passes)/nineties_played
            players.loc[players.playerId == player, 'hr_pass_suc_p90'] = hr_passes.accurate.sum()/nineties_played
            players.loc[players.playerId == player, 'hr_xP_suc_p90'] = hr_passes.xPS.sum()/nineties_played
            players.loc[players.playerId == player, 'hr_xGA_p90'] = hr_passes.xG_added.sum()/nineties_played

        hxGA_passes = pp[pp.xG_added > h_xGA_threshold]
        
        if len(hxGA_passes) > 0:
            
            players.loc[players.playerId == player, 'hxGA_passes'] = len(hxGA_passes)
            players.loc[players.playerId == player, 'hxGA_pass_suc'] = hxGA_passes.accurate.sum()/len(hxGA_passes)
            players.loc[players.playerId == player, 'hxGA_xP_suc'] = hxGA_passes.xPS.sum()/len(hxGA_passes)
            players.loc[players.playerId == player, 'hxGA_xGA'] = hxGA_passes.xG_added.sum()
            
            # Per 90.
            players.loc[players.playerId == player, 'hxGA_passes_p90'] = len(hxGA_passes)/nineties_played
            players.loc[players.playerId == player, 'hxGA_pass_suc_p90'] = hxGA_passes.accurate.sum()/nineties_played
            players.loc[players.playerId == player, 'hxGA_xP_suc_p90'] = hxGA_passes.xPS.sum()/nineties_played
            players.loc[players.playerId == player, 'hxGA_xGA_p90'] = hxGA_passes.xG_added.sum()/nineties_played
  
    players['xP_perf_p90'] = (1-(players.pass_suc_p90/players.xP_suc_p90))*100
    players['xP_F3_perf_p90'] = (1-(players.pass_suc_F3_p90/players.xP_suc_F3_p90))*100
  
    return players


def plotRadars(player_ids, players, columns):
    
    ranges = []
    params = ["Passes",
              "Pass success",
              "Net pass progression",
              "High risk passes",
              "High risk pass success",
              "High xGA passes",
              "High xGA pass success",
              "xGA",
              "xP OP",
              "xP final 3rd OP"]
    end_note = "All units are per 90 minutes"
    
    for column in columns:
        
        min_range = np.percentile(players[column], 10)
        max_range = np.percentile(players[column], 99)
        ranges.append((min_range, max_range))
        

    for player_id in player_ids:
        
        player = players[players.playerId == player_id]
        name = player.name.values[0]
        team = player.team.values[0]
        position = player.position.values[0]
        minutes = player.minutes_played.values[0]
        matches = player.matches_played.values[0]
        
        title = dict(
            title_name = name,
            subtitle_name = team,
            title_name_2 = '2017/2018',
            subtitle_name_2 = position,
            title_fontsize = 18,
            subtitle_fontsize = 13,
            title_color = '#175379',
            subtitle_color = '#CE3B2C',
            title_color_2 = '#175379',
            subtitle_color_2 = '#CE3B2C')

        values = []
        
        for column in columns:
            
            value = player[column].values[0]
            values.append(value)

        radar = Radar()
        fig, ax = radar.plot_radar(ranges=ranges,
                                         params=params,
                                         values=values,
                                         radar_color=['#175379', '#CE2B2C'],
                                         dpi=300,
                                         title=title,
                                         endnote=end_note + "\n%d minutes/%d matches" % (minutes, matches))
        
        if SAVE_OUTPUT:
            
            fig.savefig(path.join(OUTPUT_PATH, 'radar_' + name + '.png'))


def plotVSRadars(player_ids, players, columns, figax=None):
    ranges = []
    params = ["Passes",
              "Pass success",
              "Net pass progression",
              "High risk passes",
              "High risk pass success",
              "High xGA passes",
              "High xGA pass success",
              "xGA",
              "xP OP",
              "xP final 3rd OP"]
    end_note = "All units are per 90 minutes"
    
    for column in columns:
        
        min_range = np.percentile(players[column], 10)
        max_range = np.percentile(players[column], 99)
        ranges.append((min_range, max_range))
        

    player1 = players[players.playerId == player_ids[0]]
    name1 = player1.name.values[0]
    team1 = player1.team.values[0]
    player2 = players[players.playerId == player_ids[1]]
    name2 = player2.name.values[0]
    team2 = player2.team.values[0]
    matches1 = player1.matches_played.values[0]
    matches2 = player2.matches_played.values[0]

    
    title = dict(
        title_name = name1,
        subtitle_name = team1 + " - %d matches" % matches1,
        title_name_2 = name2,
        subtitle_name_2 = team2 + " - %d matches" % matches2,
        title_fontsize = 18,
        subtitle_fontsize = 13,
        title_color = '#175379',
        subtitle_color = '#175379',
        title_color_2 = '#CE3B2C',
        subtitle_color_2 = '#CE3B2C')
    
    values1 =[]
    values2 =[]
    values = [values1, values2]    
    
    for column in columns:
        
        value = player1[column].values[0]
        values1.append(value)
        value = player2[column].values[0]
        values2.append(value)

    radar = Radar(label_fontsize=14, range_fontsize=10)
    fig, ax = radar.plot_radar(ranges=ranges,
                               params=params,
                               values=values,
                               radar_color=['#175379', '#CE2B2C'],
                               dpi=150,
                               title=title,
                               endnote=end_note,
                               compare=True,
                               figax=figax)
        
    if SAVE_OUTPUT:
        
            fig.savefig(path.join(OUTPUT_PATH, 'radar_' + name1 + '_vs_' + name2 + '.png'))



if __name__ == "__main__":
    
    
    # Be polite.
    
    if not path.exists(OUTPUT_PATH):
        
        create = input("Can I please create the sub-folder %s here?\nI will store plots and some text there. (Y/n)" % OUTPUT_PATH)
        
        if create not in ('n', 'N'):
            
            makedirs(OUTPUT_PATH)
        
        else:
            
            SAVE_OUTPUT = False
                
    viz = input("Do you want to view plots and text output in-line?")
            
    if viz in ('n', 'N'):
                
        VIZ = False
    
    # Load all event data.
    # WyScout data is from 2017-2018.
    print("...loading event data...")
    pl_data = wyScoutEvents('England')
    bl_data = wyScoutEvents('Germany')
    l1_data = wyScoutEvents('France')
    ll_data = wyScoutEvents('Spain')
    sa_data = wyScoutEvents('Italy')
    
    # Concatenate data.
    data = pd.concat([pl_data, bl_data, l1_data, ll_data, sa_data])
    
    # Fetch shots.
    raw_shots = data[data.eventId == WyScoutEvents.SHOT]
    # Train naïve xG model (theta+dist) on data set.
    print("...training xG model...")
    shots, xg_model = get_xGModel(raw_shots)
    
    # Fetch passes.
    raw_passes = data[data.eventId == WyScoutEvents.PASS]
    # Remove headers and throwins - not interested.
    raw_passes = raw_passes[raw_passes.subEventId != WyScoutSubEvents.HEAD_PASS]
    raw_passes = raw_passes[raw_passes.subEventId != WyScoutSubEvents.THROW_IN]
    
    # Pre-process passes and add derived parameters.
    print("...pre-processing passes...")
    passes = preProcessPasses(raw_passes, xg_model)
 
    if VIZ:
        plotPassHeatMaps(passes, shots, "ALL")
        plotPassHeatMaps(passes, shots, "Opposition Half")

    # Evaluate impact of variables and test models.
    # Return start location model and final model.
    print("...evaluating variables and models...")
    initial_model, final_model = evaluateVariables(passes)

    # Apply models to all data where end location is in opponents half.
    # For this exercise, we're not looking at our own half.
    oppo_passes = passes[passes.end_location_x > 50]
    print("...applying xP models...")
    sk_model = linear_model.LogisticRegression(C=10000, random_state=0, multi_class='ovr', max_iter=10000)
    sk_model.fit(oppo_passes[initial_model], oppo_passes.accurate.values)
    oppo_passes['imPS'] = sk_model.predict(oppo_passes[initial_model])
    oppo_passes['imxPS'] = sk_model.predict_proba(oppo_passes[initial_model])[:,1]
    
    sk_model.fit(oppo_passes[final_model], oppo_passes.accurate.values)
    oppo_passes['mPS'] = sk_model.predict(oppo_passes[final_model])
    oppo_passes['xPS'] = sk_model.predict_proba(oppo_passes[final_model])[:,1]
    
    if VIZ:
        
        plotModelVSModel(oppo_passes)
        
    # Load match data, player and team data.
    print("...loading match, player  & team data...")
    player_data = wyScoutPlayers()
    pl_match_data = wyScoutMatches('England')
    bl_match_data = wyScoutMatches('Germany')
    l1_match_data = wyScoutMatches('France')
    ll_match_data = wyScoutMatches('Spain')
    sa_match_data = wyScoutMatches('Italy')
    match_data = pd.concat([pl_match_data, bl_match_data, l1_match_data, ll_match_data, sa_match_data])
    team_data = wyScoutTeams()
    
    print("...setting up advanced passing data for players...")
    players = investigateData(oppo_passes, match_data, player_data, team_data)
    # Drop players with < 900 minutes of playing time.
    players = players[players.minutes_played >= 900]
    
    # Split by positions.
    defenders = players[players.position == 'Defender']
    midfielders = players[players.position == 'Midfielder']
    forwards = players[players.position == 'Forward']
    
    # Filter
    columns = ['passes_p90',
               'pass_suc_p90',
               'net_progression_p90',
               'hr_passes_p90',
               'hr_pass_suc_p90',
               'hxGA_passes_p90',
               'hxGA_pass_suc_p90',
               'xGA_p90',
               'xP_perf_p90',
               'xP_F3_perf_p90']
    
    # Look at the most extreme pass model over performers - smell test...
    pm_op_def = defenders[defenders['xP_perf_p90'] > np.percentile(defenders['xP_perf_p90'], 95)]
    pm_op_mid = midfielders[midfielders['xP_perf_p90'] > np.percentile(midfielders['xP_perf_p90'], 95)]
    pm_op_fwd = forwards[forwards['xP_perf_p90'] > np.percentile(forwards['xP_perf_p90'], 95)]

    # Look at the 95th percentile players for all columns.
    defender_ids = []
    midfielder_ids = []
    forwards_ids = []
    defenders['ranking'] = np.zeros(len(defenders))
    midfielders['ranking'] = np.zeros(len(midfielders))
    forwards['ranking'] = np.zeros(len(forwards))
    
    
    # Rank players according to our criteria.
    for col in columns:
        
        # Defenders.
        col_defender_ids = defenders[defenders[col] > np.percentile(defenders[col], 95)].playerId.to_list()
        defender_ids.extend(col_defender_ids)
        def_rank = pd.DataFrame({'playerId': defenders.sort_values(col, ascending=False).playerId})
        def_rank['ranking'] = np.arange(1, len(defenders)+1)
        defenders['ranking'] += def_rank['ranking']
        
        # Midfielders.
        col_midfielder_ids = midfielders[midfielders[col] > np.percentile(midfielders[col], 95)].playerId.to_list()
        midfielder_ids.extend(col_midfielder_ids)
        mid_rank = pd.DataFrame({'playerId': midfielders.sort_values(col, ascending=False).playerId})
        mid_rank['ranking'] = np.arange(1, len(midfielders)+1)
        midfielders['ranking'] += mid_rank['ranking']
        
        # Forwards.
        col_forward_ids = forwards[forwards[col] > np.percentile(forwards[col], 95)].playerId.to_list()
        forwards_ids.extend(col_forward_ids)
        for_rank = pd.DataFrame({'playerId': forwards.sort_values(col, ascending=False).playerId})
        for_rank['ranking'] = np.arange(1, len(forwards)+1)
        forwards['ranking'] += for_rank['ranking']
        
    
    # Filter out those who tick at least four boxes.
    defender_ids_95th, def_count = np.unique(defender_ids, return_counts=True)
    midfielder_ids_95th, mid_count = np.unique(midfielder_ids, return_counts=True)
    forward_ids_95th, fwd_count = np.unique(forwards_ids, return_counts=True)
    
    top_defs = defenders[defenders.playerId.isin(defender_ids_95th[def_count > 3])]
    top_mids = midfielders[midfielders.playerId.isin(midfielder_ids_95th[mid_count > 3])]
    top_fwds = forwards[forwards.playerId.isin(forward_ids_95th[fwd_count > 3])]
    
    eval_str = ""
    eval_str += "Top rated defenders by passing model:\n"
    eval_str += "-"*80 + "\n"
    eval_str += str(top_defs[['name', 'team', 'playerId']]) + "\n"
    eval_str += "-"*80 + "\n"
    eval_str += "Top rated midfielders by passing model:\n"
    eval_str += "-"*80 + "\n"
    eval_str += str(top_mids[['name', 'team', 'playerId']]) + "\n"
    eval_str += "-"*80 + "\n"
    eval_str += "Top rated forwards by passing model:\n"
    eval_str += "-"*80 + "\n"
    eval_str += str(top_fwds[['name', 'team', 'playerId']]) + "\n"
    eval_str += "-"*80 + "\n"
    
    
    if VIZ:
        
        print(eval_str)
        
        plotRadars(defender_ids_95th[def_count > 3 ], defenders, columns)
        plotRadars(midfielder_ids_95th[mid_count > 3 ], midfielders, columns)
        plotRadars(forward_ids_95th[fwd_count > 3 ], forwards, columns)
        
        taa_50 = passes[(passes.playerId == 346101) & (passes.end_location_x > 50)]
        kw_50 = passes[(passes.playerId == 8277) & (passes.end_location_x > 50)]
        taa = passes[(passes.playerId == 346101) ]
        kw = passes[(passes.playerId == 8277) ]

        fig = plt.figure(figsize=(14,10), dpi=150)
        gs = fig.add_gridspec(2, 2, width_ratios=[2,1])
        rad_ax = fig.add_subplot(gs[:, 0])
        kw_pm_ax = fig.add_subplot(gs[0, 1])
        taa_pm_ax = fig.add_subplot(gs[1, 1])
        grey = LinearSegmentedColormap.from_list('custom cmap', ['#DADADA', 'black'])

        plotVSRadars([8277, 346101], defenders, columns, figax=(fig,rad_ax))
        
        pitch = mpl_pitch.Pitch(pitch_type='wyscout',
                                figsize=(16,9),
                                view='half',
                                stripe=False,
                                line_zorder=2)

        pitch.draw(kw_pm_ax)
        bs_heatmap = pitch.bin_statistic(kw_50.location_x, kw_50.location_y, statistic='count', bins=(12,8))
        hm = pitch.heatmap(bs_heatmap, ax=kw_pm_ax, cmap='Blues')
        fm = pitch.flow(kw_50.location_x, kw_50.location_y, kw_50.end_location_x, kw_50.end_location_y, cmap=grey, arrow_type='scale', arrow_length=50, bins=(12,8), ax=kw_pm_ax)
   
        pitch.draw(ax=taa_pm_ax)
        bs_heatmap = pitch.bin_statistic(taa_50.location_x, taa_50.location_y, statistic='count', bins=(12,8))
        hm = pitch.heatmap(bs_heatmap, ax=taa_pm_ax, cmap='Reds')
        fm = pitch.flow(taa_50.location_x, taa_50.location_y, taa_50.end_location_x, taa_50.end_location_y, cmap=grey, arrow_type='scale', arrow_length=50, bins=(12,8), ax=taa_pm_ax)
        
        fig.tight_layout()
 

        fig2 = plt.figure(figsize=(18,10), dpi=150)
        gs = fig2.add_gridspec(2, 3, width_ratios=[2,1,1])
        rad_ax = fig2.add_subplot(gs[:, 0])
        kw_el_ax = fig2.add_subplot(gs[0, 1])
        taa_el_ax = fig2.add_subplot(gs[1, 1])
        kw_pm_ax = fig2.add_subplot(gs[0, 2])
        taa_pm_ax = fig2.add_subplot(gs[1, 2])

        plotVSRadars([8277, 346101], defenders, columns, figax=(fig2,rad_ax))
        
        pitch = mpl_pitch.Pitch(pitch_type='wyscout',
                                figsize=(16,9),
                                view='half',
                                stripe=False,
                                line_zorder=2)

        pitch.draw(kw_pm_ax)
        bs_heatmap = pitch.bin_statistic(kw_50.location_x, kw_50.location_y, statistic='count', bins=(12,8))
        hm = pitch.heatmap(bs_heatmap, ax=kw_pm_ax, cmap='Blues')
        fm = pitch.flow(kw_50.location_x, kw_50.location_y, kw_50.end_location_x, kw_50.end_location_y, cmap=grey, arrow_type='scale', arrow_length=50, bins=(12,8), ax=kw_pm_ax)
   
        pitch.draw(ax=taa_pm_ax)
        bs_heatmap = pitch.bin_statistic(taa_50.location_x, taa_50.location_y, statistic='count', bins=(12,8))
        hm = pitch.heatmap(bs_heatmap, ax=taa_pm_ax, cmap='Reds')
        fm = pitch.flow(taa_50.location_x, taa_50.location_y, taa_50.end_location_x, taa_50.end_location_y, cmap=grey, arrow_type='scale', arrow_length=50, bins=(12,8), ax=taa_pm_ax)
        
        pitch.draw(kw_el_ax)
        pitch.hexbin(kw_50.end_location_x, kw_50.end_location_y, gridsize=50, ax=kw_el_ax, cmap='Blues')
        pitch.draw(taa_el_ax)
        pitch.hexbin(taa_50.end_location_x, taa_50.end_location_y, gridsize=50, ax=taa_el_ax, cmap='Reds')
        
        
        if SAVE_OUTPUT:
        
            with open(path.join(OUTPUT_PATH, 'TopRankedPlayers.txt'), 'w', encoding="utf-8") as ranked_file:
                
                ranked_file.write(eval_str)
            
            fig.savefig(path.join(OUTPUT_PATH, 'KW_vs_TAA_v1.png')) 
            fig2.tight_layout()
            fig2.savefig(path.join(OUTPUT_PATH, 'KW_vs_TAA_v2.png'))



        
        