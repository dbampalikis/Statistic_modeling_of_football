#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 00:14:22 2020

@author: dimitris
"""
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import matplotlib.animation as animation

def plot_frame( hometeam, awayteam, figax=None, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False ):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    
    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    for team,color in zip( [hometeam,awayteam], team_colors) :
        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
        ax.plot( team[x_columns], team[y_columns], color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            [ ax.text( team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 
    # plot ball
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
    return fig,ax


def to_metric_coordinates(data,field_dimen=(106.,68.) ):
    '''
    Convert positions from Metrica units to meters (with origin at centre circle)
    '''
    x_columns = [c for c in data.columns if c[-1].lower()=='x']
    y_columns = [c for c in data.columns if c[-1].lower()=='y']
    data[x_columns] = ( data[x_columns]-0.5 ) * field_dimen[0]
    data[y_columns] = 1 - ( data[y_columns]-0.5 ) * field_dimen[1]
    ''' 
    ------------ ***NOTE*** ------------
    Metrica actually define the origin at the *top*-left of the field, not the bottom-left, as discussed in the YouTube video. 
    I've changed the line above to reflect this. It was originally:
    data[y_columns] = ( data[y_columns]-0.5 ) * field_dimen[1]
    ------------ ********** ------------
    '''
    return data


def plot_pitch( field_dimen = (106.0,68.0), field_color ='green', linewidth=2, markersize=20):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1] 
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=linewidth)
    ax.plot(-x,y,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax


def calc_player_velocities(team, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocities(team)
    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()
    
    # index of first frame in second half
    second_half_idx = team.Period.idxmax(2)
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[player+"_x"].diff() / dt
        vy = team[player+"_y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        team[player + "_vx"] = vx
        team[player + "_vy"] = vy
        team[player + "_speed"] = np.sqrt( vx**2 + vy**2 )

    return team


def remove_player_velocities(team):
    # remove player velocoties and acceleeration measures that are already in the 'team' dataframe
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration']] # Get the player ids
    team = team.drop(columns=columns)
    return team


def to_single_playing_direction(home,away):
    '''
    Flip coordinates in second half so that each team always shoots in the same direction through the match.
    '''
    for team in [home,away]:
        second_half_idx = team.Period.idxmax(2)
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home,away


def calc_player_acceleration(team, player_ids):
    max_acceleration = 7
    tmp = pd.DataFrame(index=team.index)
    dt = team['Time [s]'].diff()
    for player in player_ids: # cycle through players individually
        accel = team[player+"_speed"].diff() / dt
        tmp[player + "_accel"] = accel
        tmp[tmp[player + "_accel"] > max_acceleration] = np.nan
    for player in player_ids:
        team[player + "_accel"] = tmp[player + "_accel"].rolling(5).mean()
    
    return team


def calc_goal_distance(team, pitch_length, pitch_width, left_to_right=1):
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home', 'Away'] ] )
    
    for player in player_ids: # cycle through players individually
        goal_distance = np.sqrt((pitch_length/2 + team[player+"_x"]*left_to_right)**2 + team[player+"_y"]**2)
        team[player + "_d"] = goal_distance
    return team


def calc_min_distance(team, player_ids, opponent=None, player_ids_opponent=None):
    if opponent is None:
        for player in player_ids_home:
            tmp = pd.DataFrame(index=team.index)
            for other in player_ids_home:
                if other != player:
                    tmp_distance = np.sqrt((team[player+"_x"] - team[other+"_x"])**2 + (team[player+"_y"] - team[other+"_y"])**2)
                    tmp[player+'_to_'+other] = tmp_distance
            team[player+'_near'] = tmp.min(axis=1).copy(deep=True)
            team[player+'_nearp'] = tmp.idxmin(axis=1).copy(deep=True)
    else:
        for player in player_ids_home:
            tmp = pd.DataFrame(index=opponent.index)
            for other in player_ids_opponent:
                tmp_distance = np.sqrt((team[player+"_x"] - opponent[other+"_x"])**2 + (team[player+"_y"] - opponent[other+"_y"])**2)
                tmp[player+'_to_'+other] = tmp_distance
            opponent[player+'_near'] = tmp.min(axis=1).copy(deep=True)
            opponent[player+'_nearp'] = tmp.idxmin(axis=1).copy(deep=True)
        

def get_players_positions(file_location, positions_dict, half, team_home_jerseys, team_away_jerseys):
    with open(tracks_dir) as json_data:
        data = json.load(json_data)

    for frame in range(len(data)):
        # Get match time (milliseconds) and utc time (timezone)
        positions_dict['Time [s]'].append(data[frame]['match_time']/1000)
        positions_dict['Utc_time'].append(data[frame]['utc_time'])
        positions_dict['Period'].append(half)
        #Get position of ball for each frame
        if data[frame]['ball'].get('position',np.asarray([np.inf,np.inf,np.inf])) is None:
            positions_dict['ball_x'].append(np.nan)
            positions_dict['ball_y'].append(np.nan)
        else:
            #positions_dict['ball_x'].append((data[frame]['ball']['position'][0]+pitch_length/2))
            #positions_dict['ball_y'].append((data[frame]['ball']['position'][1]+pitch_width/2))
            positions_dict['ball_x'].append((data[frame]['ball']['position'][0]+pitch_length/2)/pitch_length)
            positions_dict['ball_y'].append((data[frame]['ball']['position'][1]+pitch_width/2)/pitch_width)
    
        # Get players position for home team
        players_proccessed_home = []
        for player in range(len(data[frame]['home_team'])):
            try:
                jersey_player = data[frame]['home_team'][player]['jersey_number']
                #positions_dict['home_'+str(jersey_player)+'_x'].append((data[frame]['home_team'][player]['position'][0]+pitch_length/2))
                #positions_dict['home_'+str(jersey_player)+'_y'].append((data[frame]['home_team'][player]['position'][1]+pitch_width/2))
                positions_dict['Home_'+str(jersey_player)+'_x'].append((data[frame]['home_team'][player]['position'][0]+pitch_length/2)/pitch_length)
                positions_dict['Home_'+str(jersey_player)+'_y'].append((data[frame]['home_team'][player]['position'][1]+pitch_width/2)/pitch_width)
                players_proccessed_home.append(jersey_player)
            except:
                pass
        # Add nan for the players that are not recorded/not in the pitch for the home team
        for cur_player in team_home_jerseys:
            if cur_player not in players_proccessed_home:
                positions_dict['Home_'+str(cur_player)+'_x'].append(np.nan)
                positions_dict['Home_'+str(cur_player)+'_y'].append(np.nan)
    
        # Get players position for away team
        players_proccessed_away = []
        for player in range(len(data[frame]['away_team'])):
            try:
                jersey_player = data[frame]['away_team'][player]['jersey_number']
                #positions_dict['away_'+str(jersey_player)+'_x'].append((data[frame]['away_team'][player]['position'][0]+pitch_length/2))
                #positions_dict['away_'+str(jersey_player)+'_y'].append((data[frame]['away_team'][player]['position'][1]+pitch_width/2))
                positions_dict['Away_'+str(jersey_player)+'_x'].append((data[frame]['away_team'][player]['position'][0]+pitch_length/2)/pitch_length)
                positions_dict['Away_'+str(jersey_player)+'_y'].append((data[frame]['away_team'][player]['position'][1]+pitch_width/2)/pitch_width)
                players_proccessed_away.append(jersey_player)
            except:
                pass
        # Add nan for the players that are not recorded/not in the pitch for the away team
        for cur_player in team_away_jerseys:
            if cur_player not in players_proccessed_away:
                positions_dict['Away_'+str(cur_player)+'_x'].append(np.nan)
                positions_dict['Away_'+str(cur_player)+'_y'].append(np.nan)
    

def default_model_params(time_to_control_veto=3):
    """
    default_model_params()
    
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 12. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params


def initialise_players(team,teamname,params,GKid):
    """
    initialise_players(team,teamname,params)
    
    create a list of player objects that holds their positions and velocities from the tracking data dataframe 
    
    Parameters
    -----------
    
    team: row (i.e. instant) of either the home or away team tracking Dataframe
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returns
    -----------
    
    team_players: list of player objects for the team at at given instant
    
    """    
    # get player  ids
    player_ids = np.unique( [ c.split('_')[1] for c in team.keys() if c[:4] == teamname ] )
    # create list
    team_players = []
    for p in player_ids:
        # create a player object for player_id 'p'
        #if team[teamname+'_'+p+'_x'].isnull():
        #    continue
        team_player = player(p,team,teamname,params,GKid)
        if team_player.inframe:
            team_players.append(team_player)
    return team_players


class player(object):
    """
    player() class
    
    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player
    
    __init__ Parameters
    -----------
    pid: id (jersey number) of player
    team: row of tracking data for team
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    

    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    
    """
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self,pid,team,teamname,params,GKid):
        self.id = pid
        self.is_gk = self.id == GKid
        self.teamname = teamname
        self.playername = "%s_%s_" % (teamname,pid)
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.PPCF = 0. # initialise this for later
        
    def get_position(self,team):
        self.position = np.array( [ team[self.playername+'x'], team[self.playername+'y'] ] )
        self.inframe = not np.any( np.isnan(self.position) )
        
    def get_velocity(self,team):
        self.velocity = np.array( [ team[self.playername+'vx'], team[self.playername+'vy'] ] )
        if np.any( np.isnan(self.velocity) ):
            self.velocity = np.array([0.,0.])
    
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f


def generate_pitch_control_for_event(pass_frame, attack_location_x, attack_location_y, tracking_home, tracking_away, params, GK_numbers, attacking_team='Home', field_dimen = (106.,68.,), n_grid_cells_x = 50, offsides=True):
    """ generate_pitch_control_for_event
    
    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.
        
    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')
        
    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """

    ball_start_pos = np.array([attack_location_x, attack_location_y])
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if attacking_team=='Home':
        attacking_players = initialise_players(tracking_home.loc[pass_frame],'Home',params,GK_numbers[0])
        defending_players = initialise_players(tracking_away.loc[pass_frame],'Away',params,GK_numbers[1])
    elif attacking_team=='Away':
        defending_players = initialise_players(tracking_home.loc[pass_frame],'Home',params,GK_numbers[0])
        attacking_players = initialise_players(tracking_away.loc[pass_frame],'Away',params,GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"
        
    # find any attacking players that are offside and remove them from the pitch control calculation
    #if offsides:
    #    attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PPCFa[i,j],PPCFd[i,j] = calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params)
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    return PPCFa,xgrid,ygrid
 
   
def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ calculate_pitch_control_at_target
    
    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
    
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        return 0., 1.
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        return 1., 0.
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1]
 
    
def plot_pitchcontrol_for_event(pass_frame,  tracking_home, tracking_away, PPCF, pass_team='Home', alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (106.0,68)):
    """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )
    
    Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    NB: this function no longer requires xgrid and ygrid as an input
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """    

    # pick a pass at which to generate the pitch control surface
    #pass_frame = events.loc[event_id]['Start Frame']
    #pass_team = events.loc[event_id].Team
    
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    #plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
    
    # plot pitch control surface
    if pass_team=='Home':
        cmap = 'bwr'
    else:
        cmap = 'bwr_r'
    ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

    return fig,ax


def save_match_clip(hometeam,awayteam, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7):
    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( hometeam.index==awayteam.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = hometeam.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [hometeam.loc[i],awayteam.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
                objs, = ax.plot( team[x_columns], team[y_columns], color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
            # plot ball
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
            figobjs.append(objs)
            # include match time at the top
            frame_minute =  int( team['Time [s]']/60. )
            frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)   



file_name = '20190722.Hammarby-IFElfsborg'
#file_name = '20191020.Hammarby-MalmöFF'
#file_name = '20190930.Hammarby-Örebrö'
data_file_name=file_name+'.1'
data_file_name_second = file_name+'.2'

raw_dir = 'Signality/2019/Tracking Data/'
tracks_dir = raw_dir+data_file_name+'-tracks.json'
tracks_dir_second = raw_dir+data_file_name_second+'-tracks.json'
info_dir = raw_dir+data_file_name+'-info_live.json'


# Get the jerseys of all players for home and away teams
with open(info_dir) as json_data:
    info = json.load(json_data)

team_home_jerseys = [] 
team_away_jerseys = []

for player in info['team_home_players']:
        team_home_jerseys.append(player['jersey_number'])
for player in info['team_away_players']:
        team_away_jerseys.append(player['jersey_number'])

# Create dictionary for storing players positions and other information
positions_dict = {}
for jersey in team_home_jerseys:
    positions_dict['Home_'+str(jersey)+'_x'] = []
    positions_dict['Home_'+str(jersey)+'_y'] = []
for jersey in team_away_jerseys:
    positions_dict['Away_'+str(jersey)+'_x'] = []
    positions_dict['Away_'+str(jersey)+'_y'] = []
positions_dict['ball_x'] = []
positions_dict['ball_y'] = []
positions_dict['Time [s]'] = []
positions_dict['Utc_time'] = []
positions_dict['Period'] = []

# Get pitch dimensions
pitch_length = info['calibration']['pitch_size'][0]
pitch_width = info['calibration']['pitch_size'][1]

# Get players positions for each half
get_players_positions(tracks_dir, positions_dict, 1, team_home_jerseys, team_away_jerseys)
get_players_positions(tracks_dir_second, positions_dict, 2, team_home_jerseys, team_away_jerseys)

# Store positions dictionary to dataframe
# Transform to metric coordinates and separate home and away teams
data_df = pd.DataFrame(positions_dict)
data_df_metric = to_metric_coordinates(data_df, field_dimen=(pitch_length, pitch_width))
tracking_home = data_df_metric.filter([col for col in data_df if 'Away' not in col])
tracking_away = data_df_metric.filter([col for col in data_df if 'Home' not in col])

# Defining the goal frames, extracted manually from the videos
goal_frames = {}
goal_frames['20190722.Hammarby-IFElfsborg'] = [(26250, 26875), (44625, 45250), (57250, 57550), (118625, 119300)]
goal_frames['20191020.Hammarby-MalmöFF'] = [(20125, 21075)]
goal_frames['20190930.Hammarby-Örebrö'] = [(57625, 58375)]


player_ids_home = np.unique( [ c[:-2] for c in tracking_home.columns if c[:4] in ['Home','Away'] ] )
player_ids_away = np.unique( [ c[:-2] for c in tracking_away.columns if c[:4] in ['Home','Away'] ] )
# reverse direction of play in the second half so that home team is always attacking from right->left
tracking_home,tracking_away = to_single_playing_direction(tracking_home,tracking_away)


# Find goalkeepers' positions to figure out direction of attack
start_frame = 10
home_gk_number = info['team_home_lineup']['1']
away_gk_number = info['team_away_lineup']['1']
if tracking_home.iloc[start_frame]['Home_'+ str(home_gk_number) +'_x'] > 0: 
    tracking_home = calc_goal_distance(tracking_home, pitch_length, pitch_width, left_to_right=1)
    tracking_away = calc_goal_distance(tracking_away, pitch_length, pitch_width, left_to_right=-1)
else:
    tracking_home = calc_goal_distance(tracking_home, pitch_length, pitch_width, left_to_right=-1)
    tracking_away = calc_goal_distance(tracking_away, pitch_length, pitch_width, left_to_right=1)


# Calculate velocities and acceleration for each teams
tracking_home = calc_player_velocities(tracking_home,smoothing=True)
tracking_away = calc_player_velocities(tracking_away,smoothing=True)
tracking_home = calc_player_acceleration(tracking_home, player_ids_home)
tracking_away = calc_player_acceleration(tracking_away, player_ids_away)


# Plot any given frame
frame_to_plot = 26300
fig,ax = plot_frame(tracking_home.loc[frame_to_plot], tracking_away.loc[frame_to_plot], field_dimen = (pitch_length, pitch_width), include_player_velocities=True, annotate=True)

# Plot multiple frames
'''
#Uncomment to plot multiple frames from a match
#Some interesting sequences containing a build up to a goal are included
#  Elfsborg
plot_frames = [26550, 26675, 26725, 26800]
# Malmo
plot_frames = [20950, 20975, 20100, 20125]

for fr in plot_frames:
    fig,ax = plot_frame(tracking_home.loc[fr], tracking_away.loc[fr], field_dimen = (pitch_length, pitch_width), include_player_velocities=True, annotate=True)
'''


# Extract playes and define variables of interest
attacking_players = []
for player in range(9,12):
    attacking_players.append('Home_'+str(info['team_home_lineup'][str(player)])+'_')
plotting_variables = ['accel', 'speed', 'd']
plotting_variable_labels = ['Acceleration', 'Speed', 'Distance from goal']


# Plotting goal distance, speed and acceleration for 15 seconds (375 frames)
# Define the start of the frame to plot in frame_plot variable
frame_plot = 26500
frame_window = 375
# Extract acceleration, speed and distance from goal using 1 second window frame
plotting_df = pd.DataFrame(index=tracking_home.index)
plotting_df = tracking_home.loc[frame_plot:frame_plot+frame_window][[c for c in tracking_home.columns if c.startswith(tuple(attacking_players)) and c.endswith(tuple(plotting_variables)) ]].copy(deep=True)
plotting_mean_df = pd.DataFrame(index=range(1,16))
for pl in attacking_players:
    for pv in plotting_variables:
        plotting_df[pl+pv+'_mean'] = plotting_df[pl+pv].rolling(25).mean()
        plotting_mean_df[pl+pv] = plotting_df.iloc[25::25, :][pl+pv+'_mean'].tolist()

# Plot the variables for the 3 players
fig, axs = plt.subplots(3)
i = 0
for pv in plotting_variables:
    for pl in attacking_players:
        axs[i].plot(plotting_mean_df.index, plotting_mean_df[pl+pv], label=pl.split('_')[1])
    axs[i].set_xlabel('time (s)')
    axs[i].legend(loc='upper right')
    axs[i].set_xticks(ticks=range(1,16))
    axs[i].set_xticklabels(range(1,16))
    axs[i].set_ylabel(plotting_variable_labels[i])
    i = i+1
fig.suptitle('Goal distance, speend and acceleration over 15 seconds', y=0.92)


'''
# This code can be used for identifying errors in the data
# In this example, jumps of a specific player in the pitch
tracking_errors = pd.DataFrame(index=tracking_home.index)
for player in player_ids_home: # cycle through players individually
    dx = tracking_home[player+"_x"].diff()
    tracking_errors[player] = dx
tracking_errors[tracking_errors['Home_7'] > 1]
plot_frames = [96401, 96402]
for fr in plot_frames:
    fig,ax = plot_frame(tracking_home.loc[fr], tracking_away.loc[fr], field_dimen = (pitch_length, pitch_width), include_player_velocities=True, annotate=True)
'''

# Create dataframes for the goals scored. Define starting and ending frame number
goal_start_frame = goal_frames['20190722.Hammarby-IFElfsborg'][0][0]
goal_end_frame = goal_frames['20190722.Hammarby-IFElfsborg'][0][1]

#goal_start_frame = goal_frames['20191020.Hammarby-MalmöFF'][0][0]
#goal_end_frame = goal_frames['20191020.Hammarby-MalmöFF'][0][1]

# Plots the start and end frame. Used for figuring out if the start and end frames are correct
#fig,ax = plot_frame(tracking_home.loc[goal_start_frame], tracking_away.loc[goal_start_frame], field_dimen = (pitch_length, pitch_width), annotate=True)
#fig,ax = plot_frame(tracking_home.loc[goal_end_frame], tracking_away.loc[goal_end_frame], field_dimen = (pitch_length, pitch_width), annotate=True)

# Create dataframes for storing the distance between Hammarby's player and their teammates, as well as opponents
distances_home = tracking_home.iloc[goal_start_frame:goal_end_frame][[ c for c in tracking_home.columns if c[-2:] in ['_x','_y'] ]]
distances_away = tracking_away.iloc[goal_start_frame:goal_end_frame][[ c for c in tracking_away.columns if c[-2:] in ['_x','_y'] ]]

# Calculate min distances for each player
calc_min_distance(distances_home, player_ids_home, None, None)
calc_min_distance(distances_home, player_ids_home, distances_away, player_ids_away)



# Find all attacking players in the strating line up
attacking_players = []
for current_player in range(7,12):
    attacking_players.append('Home_'+str(info['team_home_lineup'][str(current_player)])+'_near')

# Plot the minimum distance for the players extracted
plt.rc('grid', linestyle="-", color='black')
fig, axs = plt.subplots(5)
i = 0
for pl in attacking_players:
    axs[i].plot(distances_home.index, distances_home[pl], label='Teammate')
    axs[i].plot(distances_away.index, distances_away[pl], label='Opponent')
    axs[i].set_xlabel('Frame')
    axs[i].set_ylabel('Pl ' + pl.split('_')[1])
    axs[i].legend(loc='upper right')
    axs[i].set_yticks(ticks=range(0,20,5))
    i = i+1
for ax in axs.flat:
    ax.label_outer()
fig.suptitle('Distance from other players', y=0.92)
fig.show()

# Plot the minimum distance for specific player
selected_player = 'Home_40'
plt.rc('grid', linestyle="-", color='black')
fig, axs = plt.subplots(3)
i = 0
y_labels = ['Goal 1', 'Goal 3', 'Goal 4']
for frames in goal_frames['20190722.Hammarby-IFElfsborg'][0:3]:
    distances_home = tracking_home.iloc[frames[0]:frames[1]][[ c for c in tracking_home.columns if c[-2:] in ['_x','_y'] ]]
    distances_away = tracking_away.iloc[frames[0]:frames[1]][[ c for c in tracking_away.columns if c[-2:] in ['_x','_y'] ]]
    calc_min_distance(distances_home, [selected_player], None, None)
    calc_min_distance(distances_home, [selected_player], distances_away, player_ids_away)
    axs[i].plot(distances_home.index, distances_home[selected_player+'_near'], label='Teammate')
    axs[i].plot(distances_away.index, distances_away[selected_player+'_near'], label='Opponent')
    axs[i].legend(loc='upper right')
    axs[i].set_xlabel('Frame')
    axs[i].set_ylabel(y_labels[i])
    i = i+1
fig.suptitle('Distance from other players for Durdić \n Match vs Elfsborg', y=0.96)


# PITCH CONTROL
#goal_start_frame = 20950 # Malmo 1st goal
#goal_start_frame = 57370 # Elfsborg 4th goal
goal_start_frame = 26800 # Elfsborg 1st goal

# Get ball location and goalkeeper's number on the specific frame
attack_location_x = tracking_home.iloc[goal_start_frame].ball_x
attack_location_y = tracking_home.iloc[goal_start_frame].ball_y
GK_numbers = (home_gk_number, away_gk_number)

# Define parameters for pitch control model
params = default_model_params()

# Run the pitch control model and plot the outcome
PPCF,xgrid,ygrid = generate_pitch_control_for_event(goal_start_frame, attack_location_x, attack_location_y, tracking_home, tracking_away, 
                                                    params, GK_numbers, field_dimen = (pitch_length,pitch_width,), n_grid_cells_x = 50, offsides=False)
plot_pitchcontrol_for_event(goal_start_frame, tracking_home, tracking_away, PPCF, pass_team='Home', 
                            alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (106.0,68))


'''
# Additional code for creating video clips of the various frames
# Used to figure out the correct frames for each case
video_dir = 'VIDEO/DIR/HERE'
save_match_clip(tracking_home.iloc[88530:88530+500],tracking_away.iloc[88530:88530+500],video_dir,fname='home_goal_2',include_player_velocities=False)
save_match_clip(tracking_home.iloc[goal_frames['20190722.Hammarby-IFElfsborg'][0][0]:goal_frames['20190722.Hammarby-IFElfsborg'][0][1]],
                tracking_away.iloc[goal_frames['20190722.Hammarby-IFElfsborg'][0][0]:goal_frames['20190722.Hammarby-IFElfsborg'][0][1]],
                video_dir,fname='HAM-IFE-1st-goal',include_player_velocities=False)
'''