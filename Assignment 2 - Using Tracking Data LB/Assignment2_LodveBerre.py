# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:54:17 2020

@author: lodve

This is a graded exercise.
Deadline is strict and is midnight Sunday 11th of October.

1:
The data analyst at you club has just received tracking data for three matches 
of her club (Hammarby, playing in Sweden). She wants to know what she can 
reliably infer from this data. She first asks you to calculate speed and 
direction of all players in the data using the techniques outlined in Laurie's 
videos. So, for any given frame you can plot the direction and speed of the 
players.  Give examples of 4 such frames from the matches. (3 points)

2:
Write a program which plots distance from goal, speed and acceleration of all 
Hammarby players as a short time series, i.e plot of these variables over time.
The analyst is aware there are limitations and errors in the data. Provide her
with a visualisation of the three variables for a few different examples of a 
15 second sequence of play. At least one of these should show whee the tracking
data is relatively accurate and one should show a case whee there is an error. 
Write a short report using these figures on the possibilities and limitations 
of the data. (2 points)

3:
Now the analyst now asks you to analyse Hammarby's goals. She would like you to
look at distance to nearest opposition player for each of the Hammarby players 
during the possession (you will have to find the start of possession time frame
by hand) leading up to an open play goal. Measure the distance to the nearest 
defending player and nearest teammate during  this buildup. Use this to a
nalyse one Hammarby player in particular. (3 points)

4:
Implement a further metric of you own choice that you think will help 
understand how Hammarby score goals. Use this to analyse your chosen Hammarby 
player.  (2 points)

Submission should consist of 2 parts.

1: A two page document containing:

* a non-technical description of how your method for measuring distance, speed,
  acceleration and relative positions of other players works.
* the examples of frames with speed and direction and time series of these 
  variables for short sequences.
* an explanation of the strengths and weaknesses of the data as asked for by 
  the data analyst.
* an analysis of the attacking runs of one of the Hammarby players in 
  particular.

2:
A runnable, commented code  as a (preferably) Python or R script that generates
all the plots from the report and explains the method you have used. 
Important: this code should be a single file run immediately if placed with in
the same directory as the Signality folder.
"""

from datetime import datetime
from glob import glob
from mplsoccer import pitch as mpl_pitch
from os import path
from scipy.spatial import cKDTree

import cycler
import json
import matplotlib.animation as animation
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd

SIGNALITY_PATH = '2019/Tracking Data'
DEBUG = False

###############################################################################
### -------------------- Start Methods for loading data ------------------- ###
###############################################################################

class TeamMap:
    
    HAM = 'Hammarby'
    IFE = 'IFElfsborg'
    ORE = 'Örebro'
    MFF = 'MalmöFF'


class MatchInfo:
    """
    Class used to store match information.
    Provides easy access to pitch size and team information.
    """
    
    pitchSize = None
    homeTeamName = None
    homeTeamLineup = None
    awayTeamName = None
    awayTeamLineup = None
    
    def __init__(self, info):
        """
        Constructor.

        Parameters
        ----------
        info : dict
            Raw dict data as read from the info json file.

        Returns
        -------
        None.

        """
        
        self.pitchSize = info['calibration']['pitch_size']
        self.startTime = np.datetime64(datetime.strptime(info['time_start'], "%Y-%m-%dT%H:%M:%S.%fZ"))
        self.homeTeamName = info['team_home_name']
        home_lu = pd.Series(info['team_home_lineup'])
        home_pl = pd.json_normalize(info['team_home_players'])
        home_pl['position'] = np.zeros(len(home_pl))
        home_pl['position'] = home_pl['position'].astype(int)
        home_pl.loc[home_pl.jersey_number.isin(home_lu), 'position'] = home_lu.index
        self.homeTeamLineup = home_pl
        
        self.awayTeamName = info['team_away_name']
        self.awayTeamLineup = pd.json_normalize(info['team_away_players'])
        away_lu = pd.Series(info['team_away_lineup'])
        away_pl = pd.json_normalize(info['team_away_players'])
        away_pl['position'] = np.zeros(len(away_pl))
        away_pl.loc[away_pl.jersey_number.isin(away_lu), 'position'] = away_lu.index
        self.awayTeamLineup = away_pl
    
    def __repr__(self):

        repr_str = "Match info for:\n{} - {}\n\n".format(self.homeTeamName, self.awayTeamName)
        home_11 = self.homeTeamLineup[self.homeTeamLineup.position != 0]
        away_11 = self.awayTeamLineup[self.awayTeamLineup.position != 0]
        
        for h_n, h_j, a_n, a_j in zip(home_11.name.values, home_11.jersey_number.values, away_11.name.values, away_11.jersey_number.values):
            
            repr_str += "%s (%d)          %s (%d)\n" % (h_n, h_j, a_n, a_j)
            
        return repr_str

        
    def __str__(self):
        
        return self.__str__()



def getSignalityData(home, away, half=None):
    """
    Load Signality match data.
    Events, tracking, info & stats.
    
    Expands json fields and returns massive dataframes.

    Parameters
    ----------
    home : string
        Home team name as stored to disk.
    away : string
        Away team name.
    half : Int
        Which half to load data for. If None, load all.

    Returns
    -------
    Pandas Dataframes, dict
        DESCRIPTION.

    """
    
    def addPlayerColumns(tracks, info):
        """
        Add one column for each player in lineup.
        Fill with NaNs.

        Parameters
        ----------
        tracks : Pandas DataFrame
            Dataframe with virgin tracking info loaded from json file.
        info : MatchInfo
            Match meta information.

        Returns
        -------
        tracks : Pandas DataFrame
            Tracking data DataFrame with NaN filled columns for all players.
        """
        
        home_cols = ["home_%d_pos" % j_no for j_no in info.homeTeamLineup.jersey_number.values]
        home_vals = np.full((len(tracks), len(home_cols)), None, dtype=object)
        tracks[home_cols] = home_vals
        home_data = tracks.home_team.to_list()

        # Extract positional data from dicts to columns.
        for i, player_dicts in enumerate(home_data):
            
            for player_dict in player_dicts:
                jersey = player_dict.get('jersey_number')
                player = "home_%d_pos" % jersey
                pos = player_dict.get('position')
                tracks.at[i, player] = pos
                
        # Split positional data into separate columns for x & y.
        for col in home_cols:
            
            # Skip columns for players without values.
            if tracks[col].isna().sum() == len(tracks):
                
                print("WARNING: No values found for {}".format(col))
                continue
            
            x = col + "_x"
            y = col + "_y"
            
            tracks[[x, y]] = pd.DataFrame(tracks[tracks[col].notna()][col].to_list(), index=tracks[tracks[col].notna()].index)
            
        tracks.drop(columns=home_cols, inplace=True)

        away_cols = ["away_%d_pos" % j_no for j_no in info.awayTeamLineup.jersey_number.values]
        away_vals = np.full((len(tracks), len(away_cols)),None, dtype=object)
        tracks[away_cols] = away_vals
        away_data = tracks.away_team.to_list()

        # Extract positional data from dicts to columns.
        for i, player_dicts in enumerate(away_data):
            
            for player_dict in player_dicts:
                jersey = player_dict.get('jersey_number')
                player = "away_%d_pos" % jersey
                pos = player_dict.get('position')
                tracks.at[i, player] = pos
        
        # Split positional data into separate columns for x & y.
        for col in away_cols:
            
            # Skip columns for players without values.
            if tracks[col].isna().sum() == len(tracks):
                
                print("WARNING: No values found for {}".format(col))
                continue
            
            x = col + "_x"
            y = col + "_y"
            
            tracks[[x, y]] = pd.DataFrame(tracks[tracks[col].notna()][col].to_list(), index=tracks[tracks[col].notna()].index)
        
        tracks.drop(columns=away_cols, inplace=True)
        
        return tracks
    
    
    sig_paths = glob(path.join(SIGNALITY_PATH, "*.{}-{}.{}-*.json".format(home, away, half)))
    
    event_path = ""
    info_path = ""
    stats_path = ""
    track_path = ""
    
    for sig_path in sig_paths:
    
        if 'events' in sig_path:
            
            event_path = sig_path
            
        elif 'info_live' in sig_path:
            
            info_path = sig_path
            
        elif 'stats' in sig_path:
            
            stats_path = sig_path
            
        elif 'tracks' in sig_path:
            
            track_path = sig_path
            
        else:
            
            print("WARNING: Found unknow data type in file {}!".format(sig_path))
    
    # For the events we expand the event types.
    events = pd.read_json(event_path)
    events_normed = pd.json_normalize(events.event)
    events = pd.concat([events, events_normed], axis=1)
    events.drop(columns=['event'], inplace=True)

    # Info can't be read directly into a dataframe.    
    with open(info_path) as info_data:
        
        info = MatchInfo(json.load(info_data))
    
    # For the stats we alse expand the stats.
    stats = pd.read_json(stats_path)
    stats_normed = pd.json_normalize(stats.stats)
    stats = pd.concat([stats, stats_normed], axis=1)
    stats.drop(columns=['stats'], inplace=True)
    
    # And finally expand all player columns as well as the ball positions.
    
    tracks = pd.read_json(track_path)
    
        
    tracks = addPlayerColumns(tracks, info)
    ball_tracks = pd.json_normalize(tracks.ball)
    tracks['ball_pos'] = ball_tracks.position
    ball_pos = pd.DataFrame(tracks[tracks.ball_pos.notna()].ball_pos.to_list(), index=tracks[tracks.ball_pos.notna()].index)
    balls = np.full((len(tracks), 3), np.nan)
    tracks[['ball_pos_x', 'ball_pos_y', 'ball_pos_z']] = balls
    tracks['ball_pos_x'] = ball_pos[0]
    tracks['ball_pos_y'] = ball_pos[1]
    tracks['ball_pos_z'] = ball_pos[2]
    tracks['ball_player'] = ball_tracks.player
    tracks.drop(columns=['ball', 'ball_pos'], inplace=True)
    
    return events, info, stats, tracks

###############################################################################
### --------------------- End Methods for loading data -------------------- ###
###############################################################################

###############################################################################
### ----------------------- Start of Physics Methods ---------------------- ###
###############################################################################

class PitchControlModel:
    """
    Pitch is divided in to X*Y pixels.
    
    For a given location on the pitch:
        1. How long will it take for the ball to arrive?
        2. How long will it take for each player to arrive?
        3. What is the total probability that each team will control the ball
           post-arrival?
        
    Simple approximation for arrival time:
        1. Initial reaction time of 0.7s - during this time player move along
           current trajectory
        2. After 0.7s player runs directly towards target location at maximum
           speed of 5 m/s.
    
    Ball control assumed as a stochastic process with a fixed rate:
       For each time interval delta_t that a player is in the vicinity of the 
       ball, (s)he has a probability of lambda*delta_t to make a contrlled
       touch on the ball.
       
    We use physics to compute the optimal time-to-intercept assuming a 
    personalized maximum speed and acceleration. This is converted into a
    sigmoid distribution with a sigma of 0.45s. This is to reflect uncertainty
    in arrival time.
    
    Assumptions:
        Players have a maximum speed of 5 m/s
        Players have a maximum acceleration of 7 m/s²
    """
    
    def __init__(self, gridX, pitchDimXY, modParams=None):
   
        if modParams is None:
            
            self.modParams = self.getDefaultModelParams()
            
        else:
            
            self.modParams = modParams
        
        grid_x = gridX
        pitch_x, pitch_y = pitchDimXY
        grid_y = int(grid_x*pitch_y/pitch_x)
        # break the pitch down into a grid
        self.x_grid = np.linspace( -pitch_x/2., pitch_x/2., grid_x)
        self.y_grid = np.linspace( -pitch_y/2., pitch_y/2., grid_y)


    def calcPCatPos(self, targetPos, attPlayers, defPlayers, ballStartPos):
        """
        Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
        Parameters
        -----------
            targetPos: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
            attPlayers: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
            defPlayers: list of 'player' objects (see player class above) for the players on the defending team
            ballStartPos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
            modParams: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
        Returns
        -----------
            PPCFatt: Pitch control probability for the attacking team
            PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

        """

        avg_ball_speed = self.modParams['average_ball_speed']
        time_to_control_att = self.modParams['time_to_control_att']
        time_to_control_def = self.modParams['time_to_control_def']
        int_dt = self.modParams['int_dt']
        max_int_time = self.modParams['max_int_time']
        model_converge_tol = self.modParams['model_converge_tol']
        lambda_att = self.modParams['lambda_att']
        lambda_def = self.modParams['lambda_def']

        # Calculate ball travel time from start position to end position.   
        if ballStartPos is None or any(np.isnan(ballStartPos)):
            
            # We here assume that ball is already at location
            ball_travel_time = 0.0 

        else:

            # Ball travel time is distance to target position from current ball
            # position divided assumed average ball speed.
            ball_travel_time = np.linalg.norm(targetPos - ballStartPos)/avg_ball_speed
    
        # First get arrival time of 'nearest' attacking player
        # (nearest also dependent on current velocity).
        tau_min_att = np.nanmin([p.timeToInterceptSimple(targetPos) for p in attPlayers])
        tau_min_def = np.nanmin([p.timeToInterceptSimple(targetPos) for p in defPlayers])
    
        # Check whether we actually need to solve equation 3
        if tau_min_att - max(ball_travel_time, tau_min_def) >= time_to_control_def:
            # If defending team can arrive significantly before attacking team, 
            # no need to solve pitch control model.
            return 0., 1.

        elif tau_min_def - max(ball_travel_time, tau_min_att) >= time_to_control_att:

            # If the attacking team can arrive significantly before defending team, 
            # no need to solve pitch control model.
            return 1., 0.

        else: 

            # Solve pitch control model by integrating equation 3 in Spearman et al.
            # First remove any player that is far (in time) from the target location
            attPlayers = [p for p in attPlayers if p.time_to_intercept-tau_min_att < time_to_control_att]
            defPlayers = [p for p in defPlayers if p.time_to_intercept-tau_min_def < time_to_control_def]
            
            # set up integration arrays
            dT_array = np.arange(ball_travel_time-int_dt, ball_travel_time+max_int_time, int_dt) 
            PPCFatt = np.zeros_like(dT_array)
            PPCFdef = np.zeros_like(dT_array)
            # Integrate equation 3 of Spearman 2018 until convergence or 
            # tolerance cut off limit hit (see 'params').
            ptot = 0.0
            i = 1

            while 1 - ptot > model_converge_tol and i < dT_array.size: 

                T = dT_array[i]

                for player in attPlayers:

                    # calculate ball control probablity for 'player' in time interval T+dt
                    dPPCFdT = (1 - PPCFatt[i-1] - PPCFdef[i-1])*player.probabilityOfBallIntercept(T)*lambda_att
                    # make sure it's greater than zero
                    assert dPPCFdT >= 0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                    player.PPCF += dPPCFdT*int_dt# total contribution from individual player
                    PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)

                for player in defPlayers:

                    # calculate ball control probablity for 'player' in time interval T+dt
                    dPPCFdT = (1 - PPCFatt[i-1] - PPCFdef[i-1])*player.probabilityOfBallIntercept(T)*lambda_def
                    # make sure it's greater than zero
                    assert dPPCFdT >= 0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                    player.PPCF += dPPCFdT*int_dt # total contribution from individual player
                    PPCFdef[i] += player.PPCF # add to sum over players in the defending team

                ptot = PPCFdef[i] + PPCFatt[i] # total pitch control probability 
                i += 1

            if i >= dT_array.size:
            
                print("Integration failed to converge: %1.3f" % (ptot) )
            
            return PPCFatt[i-1], PPCFdef[i-1]


    def initPlayers(self, player_ids, frame, teamname):
        """
        Create a list of player objects that holds their positions and velocities from the tracking data dataframe 
        
        Parameters
        -----------
        
        frame: row (i.e. instant) of the tracking Dataframe
        teamname: team name "home" or "away"
        
        Returns
        -----------
    
        team_players: list of player objects for the team at at given instant
    
        """    
        # create list
        team_players = []
 
        for p in player_ids:
 
            # create a player object for player_id 'p'
            team_player = self.Player(p, teamname, frame, self.modParams)
       
            if team_player.in_frame:

                team_players.append(team_player)
        
        return team_players
        

    def getDefaultModelParams(self, timeToControlCutOff=3):
        """        
        Returns the default parameters that define and evaluate the model.
        See Spearman (2018) for more details.
    
        Parameters
        -----------
        time_to_control_veto: If the probability that another team or player
        can get to the ball and control it is less than 10^-timeToControlCutOff,
        ignore that player.
    
    
        Returns
        -----------
        params: dictionary of parameters required to determine and calculate the model
        
        """
        # key parameters for the model, as described in Spearman 2018
        params = {}
        # model parameters
        params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
        params['max_player_speed'] = 5. # maximum player speed m/s
        params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
        params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
        params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
        params['lambda_att'] = 4.3 # ball control parameter for attacking team
        params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
        params['average_ball_speed'] = 15. # average ball travel speed in m/s
        # numerical parameters for model evaluation
        params['int_dt'] = 0.04 # integration timestep (dt)
        params['max_int_time'] = 10 # upper limit on integral time
        params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
        # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
        # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
        params['time_to_control_att'] = timeToControlCutOff*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
        params['time_to_control_def'] = timeToControlCutOff*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])

        return params


    def getGridDims(self):    

        return self.x_grid, self.y_grid
    
    
    def getPitchControlForFrame(self, frame, default='home'):
        """
        Evaluates pitch control surface over the entire field at the moment of the given frame passed as an input.
    
        Parameters
        -----------
            frame: Frame with tracking data.
            default: Team to assume in possession if no team flagged in frame.
                            
        Returns
        -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x, n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCFa.

        """ 
        # Get the details of the frame; team in possession, ball position etc.
        player_on_ball = frame['ball_player']
        home_meta = pd.json_normalize(frame['home_team'])
        away_meta = pd.json_normalize(frame['away_team'])
        
        if player_on_ball in home_meta.track_id.values:
            
            poss_team = 'home'
            # pass_player_id = home_meta[home_meta['track_id'] == player_on_ball].jersey_number.values[0]
            # pass_player = pass_team + "_%d" % pass_player_id
            
        elif player_on_ball in away_meta.track_id.values:
            
            poss_team = 'away'
            # pass_player_id = away_meta[away_meta['track_id'] == player_on_ball].jersey_number.values[0]
            # pass_player = pass_team + "_%d" % pass_player_id
            
        else:
            
            poss_team = default
        
        ball_start_pos = np.asarray([frame.ball_pos_x, frame.ball_pos_y])
        
        # initialise pitch control grids for attacking and defending teams 
        PPCFa = np.zeros(shape = (len(self.y_grid), len(self.x_grid)))
        PPCFd = np.zeros(shape = (len(self.y_grid), len(self.x_grid)))
        
        # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
        if poss_team == 'home':
        
            att_players = self.initPlayers(home_meta.jersey_number.values, frame,'home')
            def_players = self.initPlayers(away_meta.jersey_number.values, frame,'away')
        
        elif poss_team == 'away':
        
            def_players = self.initPlayers(home_meta.jersey_number.values, frame, 'home')
            att_players = self.initPlayers(away_meta.jersey_number.values, frame, 'away')

        else:

            assert False, "Team in possession must be either home or away"

        # calculate pitch pitch control model at each location on the pitch
        for i in range(len(self.y_grid)):

            for j in range(len(self.x_grid)):

                target_position = np.array([self.x_grid[j], self.y_grid[i]])
                PPCFa[i,j], PPCFd[i,j] = self.calcPCatPos(target_position, att_players, def_players, ball_start_pos)

        # check probabilitiy sums within convergence
        checksum = np.sum(PPCFa + PPCFd)/float(self.y_grid.shape[0]*self.x_grid.shape[0]) 
        assert 1-checksum < self.modParams['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
        
        return poss_team, PPCFa

    
    class Player:
        """
        Everything the pitch control model needs to know about a player.
        """
        
        def __init__(self, pid, team, frame, params):
            
            self.pid = pid
            self.frame = frame
            self.player = "%s_%s_" % (team, pid)
            self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
            self.position, self.in_frame = self._getPosition(frame, self.player)
            self.velocity, self.vmax = self._getVelocity(frame, self.player)
            self.vmax = 5
            self.PPCF = 0. # initialise this for later
            self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
            
        def _getPosition(self, frame, player):
 
            position = np.array([frame[player+'pos_x'], frame[player+'pos_y']]) 
            in_frame = not np.any(np.isnan(position))
        
            return position, in_frame

            
        def _getVelocity(self, frame, player):

            velocity = np.array([frame[player+'pos_vel'], frame[player+'pos_vel']])
            
            if np.any(np.isnan(velocity)):

                velocity = np.array([0.,0.])
                vmax = 0
                
            else:
                
                vmax = (velocity[0]**2+velocity[1]**2)**0.5

            return velocity, vmax

        
        def timeToInterceptSimple(self, r_final):
            self.PPCF = 0. # initialise this for later
            # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
            # and then runs at full speed to the target position.
            r_reaction = self.position + self.velocity*self.reaction_time
            self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
            return self.time_to_intercept

        def probabilityOfBallIntercept(self, T):
            # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
            f = 1/(1. + np.exp(-np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept)))
            return f
        

def calcVelocities(data, home_ids, away_ids, filtering=None, interp_nans=False):
    """
    Calculate velocities for all players.
    Optionally filter outliers.

    Parameters
    ----------
    data : Pandas DataFrame
        DataFrame containing player positional data.
    home_ids : [int]
        List of integers with the home player jersey numbers.
    away_ids : [int]
        List of integers with the away player jersey numbers.
    filtering : Function
        Filtering method to use. Must be able to work on 1D arrasy.
        The default is None.

    Returns
    -------
    Pandas DataFrame with player velocities added.

    """
    
    data = data.copy()
    
    # def nan_helper(y):
        
    #     return np.isnan(y), lambda z: z.nonzero()[0]
    
    usain = 12 # Usain Bolt type speed.
    usain_acc = 16 # Usain Bolt type acceleration.
    window = 7
    # p_order = 1
    players = ["home_%d_pos" % j_no for j_no in home_ids]
    players.extend(["away_%d_pos" % j_no for j_no in away_ids])

    dt = data.utc_time.diff()
    dt = dt.apply(lambda dt: dt.total_seconds())
    
    # Calculate velocities for all players.
    for player in players:
        
        x_col = player + "_x"
        y_col = player + "_y"
        v_col = player + "_vel"
        vx_col = player + "_x_vel"
        vy_col = player + "_y_vel"
        a_col = player + "_acc"
        ax_col = player + "_x_acc"
        ay_col = player + "_y_acc"

        # Skip players without data.
        if not x_col in data.columns:
            
            print("WARINING: could not find positonal data for {}, skipping!".format(player))
            continue

        if DEBUG:
            
            print("Now calculating velocity and acceleration for ", player)

        vx = data[x_col].diff()/dt
        vy = data[y_col].diff()/dt
        v = (vx**2 + vy**2)**0.5

        # For the purpose of this exercise, use Savitzky-Golay filtering.
        # If it's good enough for Laurie Shaw, it's good enough for me.
        # TODO: Make it work, Least Squares fit won't converge... :P
        # WORKAROUND: Use short moving average for now.
        if filtering:
        
            print("Now filtering velocities for player ", player)
            # First remove obvious spikes.
            vx[v > usain] = np.nan
            vy[v > usain] = np.nan
            
            ma_window = np.ones(window)/window
            vx = np.convolve(vx, ma_window, mode='same')
            vy = np.convolve(vy, ma_window, mode='same')
            
            # Interpolate over spikes.
            # nans, x = nan_helper(vx)
            # nans, y = nan_helper(vy)
            # vx[nans] = np.interp(x(nans), x(~nans), vx[~nans])
            # vy[nans] = np.interp(y(nans), y(~nans), vy[~nans])
            # vx = savgol_filter(vx, window, p_order)
            # vy = savgol_filter(vy, window, p_order)
            
            v = (vx**2 + vy**2)**0.5
            
            print("Velocities filtering successful for player ", player)
            
        ax = np.gradient(vx)/dt
        ay = np.gradient(vy)/dt
        # a = (ax**2 + ay**2)**0.5
        a = np.gradient(v)/dt
        
        if filtering:
        
            print("Now filtering acceleration for player ", player)
            # First remove obvious spikes.
            ax[a > usain_acc] = np.nan
            ay[a > usain_acc] = np.nan
            
            ma_window = np.ones(window)/window
            ax = np.convolve(ax, ma_window, mode='same')
            ay = np.convolve(ay, ma_window, mode='same')
            
            # Interpolate over spikes.
            # nans, x = nan_helper(vx)
            # nans, y = nan_helper(vy)
            # vx[nans] = np.interp(x(nans), x(~nans), vx[~nans])
            # vy[nans] = np.interp(y(nans), y(~nans), vy[~nans])
            # vx = savgol_filter(vx, window, p_order)
            # vy = savgol_filter(vy, window, p_order)
            
            # a = (ax**2 + ay**2)**0.5
            a = np.convolve(a, ma_window, mode='same')
            
            print("Acceleration filtering successful for player ", player)
        
        data[vx_col] = vx
        data[vy_col] = vy
        data[v_col] = v
    
        data[ax_col] = ax
        data[ay_col] = ay
        data[a_col] = a
        
        # Voronoi diagrams etc fails with NaNs.
        # For the purpose of this exericse, the easiest way to deal with
        # it is plain old ugly interpolation over missing values...
        if interp_nans:
            
            data[x_col].interpolate(method='linear', inplace=True)
            data[y_col].interpolate(method='linear', inplace=True)
            data[v_col].interpolate(method='linear', inplace=True)
            data[vx_col].interpolate(method='linear', inplace=True)
            data[vy_col].interpolate(method='linear', inplace=True)
            data[a_col].interpolate(method='linear', inplace=True)
            data[ax_col].interpolate(method='linear', inplace=True)
            data[ay_col].interpolate(method='linear', inplace=True)
        
        if DEBUG:
            
            print("Done with player ", player)
            
    # Calculate velocity for ball.
    # Here be dragons. Positional data much worse than player positional data.
    ball_vx = data.ball_pos_x.diff()/dt
    ball_vy = data.ball_pos_y.diff()/dt
    ball_v = (ball_vx**2+ball_vy**2)**0.5
    
    if filtering:
        
        # First remove obvious spikes.
        ball_vx[v > usain*5] = np.nan
        ball_vy[v > usain*5] = np.nan
        
        ma_window = np.ones(window)/window
        ball_vx = np.convolve(ball_vx, ma_window, mode='same')
        ball_vy = np.convolve(ball_vy, ma_window, mode='same')
        
        # Interpolate over spikes.
        # nans, x = nan_helper(ball_vx)
        # nans, y = nan_helper(ball_vy)
        # ball_vx[nans] = np.interp(x(nans), x(~nans), ball_vx[~nans])
        # ball_vy[nans] = np.interp(y(nans), y(~nans), ball_vy[~nans])
        
        # ball_vx = savgol_filter(ball_vx, window, p_order)
        # ball_vy = savgol_filter(ball_vy, window, p_order)
        ball_v = (ball_vx**2 + ball_vy**2)**0.5    
    
    data['ball_pos_x_vel'] = ball_vx
    data['ball_pos_y_vel'] = ball_vy
    data['ball_pos_vel'] = ball_v
    
    if interp_nans:
        
        data['ball_pos_x_vel'].interpolate(method='linear', inplace=True)
        data['ball_pos_y_vel'].interpolate(method='linear', inplace=True)
        data['ball_pos_vel'].interpolate(method='linear', inplace=True)
    
    return data


def getVitalityData(meta_info, frames):
    """
    
    Accelerations/decelerations are defined as being above 2 m/s^2 for > 0.5 s.
    """
    hids = meta_info.homeTeamLineup.jersey_number.values
    aids = meta_info.awayTeamLineup.jersey_number.values
    pitch_length = meta_info.pitchSize[0]
    cols = ['DistTot', 'DistWalk', 'DistJog', 'DistRun', 'DistSprint', 'DistAtt3rd', 'DistMid3rd', 'DistDef3rd', 'NoSprints', 'NoAccels', 'NoDecs']
    hind = ['home_%d' % hid for hid in hids]
    aind = ['away_%d' % aid for aid in aids]
    indx = hind.copy()
    indx.extend(aind)
    emp_data = np.full((len(hids)+len(aids), len(cols)), 0.0)
    vit_data = pd.DataFrame(data=emp_data, columns=cols, index=indx)
    att_3rd_lim = pitch_length/3-pitch_length/2
    def_3rd_lim = pitch_length/2-pitch_length/3
    
    # Get vitality data for home team.
    for hid in hids:
        
        no_sprints = 0
        no_accs = 0
        no_decs = 0
        player = "home_%d" % hid
        pos_x_col = "home_%d_pos_x" % hid
        vel_col = "home_%d_pos_vel" % hid
        acc_col = "home_%d_pos_acc" % hid
        
        if vel_col not in frames.columns:
            
            continue
        
        if DEBUG:
            
            print("Calculating vitality stats for ", player)
        
        sprints = frames[frames[vel_col] >= 7][[vel_col, 'match_time']].reset_index()
        runs = frames[(frames[vel_col] >= 4) & (frames[vel_col] < 7)][[vel_col, 'match_time']].reset_index()
        jogs = frames[(frames[vel_col] >= 2) & (frames[vel_col] < 4)][[vel_col, 'match_time']].reset_index()
        walks = frames[frames[vel_col] < 2][[vel_col, 'match_time']].reset_index()
        start_inds = (sprints[sprints.match_time.diff() > 40].index).to_list()
        end_inds = (sprints[sprints.match_time.diff() > 40].index-1).to_list()       
        att_3rd = frames[frames[pos_x_col] <= att_3rd_lim]
        mid_3rd = frames[(frames[pos_x_col] > att_3rd_lim) & (frames[pos_x_col] < def_3rd_lim)]
        def_3rd = frames[frames[pos_x_col] >= def_3rd_lim]
        
        if len(start_inds) > 0 and len(end_inds) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one sprint zone for player", player)
                
            start_inds.insert(0, 0)
            end_inds.append(len(sprints)-1)
        
        elif len(sprints) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous sprint zone for player", player)
                
            if ((sprints.iloc[-1].match_time-sprints.iloc[0].match_time)/1000 > 1):
        
                print("Found ONE sustained sprint for player", player)
                no_sprints = 1
                
        # Calculate number of sustained sprints
        if DEBUG:
            
            print("Calculating number of sprints...")
            
        for si, ei in zip(start_inds, end_inds):
            
            st = (sprints.iloc[ei].match_time - sprints.iloc[si].match_time)/1000
            
            if DEBUG:
            
                print("Sprint duration: %.2f secs" % st)
            
            if st > 1:
                
                no_sprints += 1
        
        # Sustained accelerations/decelerations.
        accs = frames[frames[acc_col] > 3][[acc_col, 'match_time']].reset_index()
        start_inds_accs = (accs[accs.match_time.diff() > 40].index).to_list()
        end_inds_accs = (accs[accs.match_time.diff() > 40].index-1).to_list()       
        decs = frames[frames[acc_col] < -3][[acc_col, 'match_time']].reset_index()
        start_inds_decs = (decs[decs.match_time.diff() > 40].index).to_list()
        end_inds_decs = (decs[decs.match_time.diff() > 40].index-1).to_list()       

        
        if len(start_inds_accs) > 0 and len(end_inds_accs) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one acceleration zone for player", player)
                
            start_inds_accs.insert(0, 0)
            end_inds_accs.append(len(accs)-1)
        
        elif len(accs) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous acceleration zone for player", player)
                
            if ((accs.iloc[-1].match_time-accs.iloc[0].match_time)/1000 > 0.5):
        
                print("Found ONE sustained acceleration for player", player)
                no_accs = 1
        
        if len(start_inds_decs) > 0 and len(end_inds_decs) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one deceleration zone for player", player)
                
            start_inds_decs.insert(0, 0)
            end_inds_decs.append(len(decs)-1)
        
        elif len(decs) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous deceleration zone for player", player)
                
            if ((decs.iloc[-1].match_time-decs.iloc[0].match_time)/1000 > 0.5):
        
                print("Found ONE sustained deceleration for player", player)
                no_decs = 1
        
        # Calculate number of sustained accelerations.
        if DEBUG:
            
            print("Calculating number of accelerations...")
        
        for si, ei in zip(start_inds, end_inds):
            
            sa = (accs.iloc[ei].match_time - accs.iloc[si].match_time)/1000
            
            if sa > 0.5:
                
                no_accs += 1
   
        # Calculate number of sustained decelerations.
        if DEBUG:
            
            print("Calculating number of decelerations...")
        
        for si, ei in zip(start_inds_decs, end_inds_decs):
            
            sd = (decs.iloc[ei].match_time - decs.iloc[si].match_time)/1000
            
            if sd > 0.5:
                
                no_decs += 1
   
    
        tot_dist_cov = frames[vel_col].sum()/25/1000 # In km.
        tot_dist_spr = sprints[vel_col].sum()/25/1000 # In km.
        tot_dist_run = runs[vel_col].sum()/25/1000
        tot_dist_jog = jogs[vel_col].sum()/25/1000
        tot_dist_wal = walks[vel_col].sum()/25/1000
        tot_dist_att_3rd = att_3rd[vel_col].sum()/25/1000
        tot_dist_mid_3rd = mid_3rd[vel_col].sum()/25/1000
        tot_dist_def_3rd = def_3rd[vel_col].sum()/25/1000

        vit_data.loc[player]['DistTot'] = tot_dist_cov
        vit_data.loc[player]['DistWalk'] = tot_dist_wal
        vit_data.loc[player]['DistJog'] = tot_dist_jog
        vit_data.loc[player]['DistRun'] = tot_dist_run
        vit_data.loc[player]['DistSprint'] = tot_dist_spr
        vit_data.loc[player]['NoSprints'] = no_sprints
        vit_data.loc[player]['NoAccels'] = no_accs
        vit_data.loc[player]['NoDecs'] = no_decs
        vit_data.loc[player]['DistAtt3rd'] = tot_dist_att_3rd
        vit_data.loc[player]['DistMid3rd'] = tot_dist_mid_3rd
        vit_data.loc[player]['DistDef3rd'] = tot_dist_def_3rd
        
    # Calculate vitality data for away team.
    att_3rd_lim = pitch_length/2-pitch_length/3
    def_3rd_lim = pitch_length/3-pitch_length/2
    
    for aid in aids:
        
        no_sprints = 0
        no_accs = 0
        no_decs = 0
        pos_x_col = "away_%d_pos_x" % aid
        player = "away_%d" % aid
        vel_col = "away_%d_pos_vel" % aid
        acc_col = "away_%d_pos_acc" % aid
        
        if vel_col not in frames.columns:
            
            continue
        
        sprints = frames[frames[vel_col] >= 7][[vel_col, 'match_time']].reset_index()
        runs = frames[(frames[vel_col] >= 4) & (frames[vel_col] < 7)][[vel_col, 'match_time']].reset_index()
        jogs = frames[(frames[vel_col] >= 2) & (frames[vel_col] < 4)][[vel_col, 'match_time']].reset_index()
        walks = frames[frames[vel_col] < 2][[vel_col, 'match_time']].reset_index()
        start_inds = (sprints[sprints.match_time.diff() > 40].index).to_list()
        end_inds = (sprints[sprints.match_time.diff() > 40].index-1).to_list()       
        att_3rd = frames[frames[pos_x_col] >= att_3rd_lim]
        mid_3rd = frames[(frames[pos_x_col] < att_3rd_lim) & (frames[pos_x_col] > def_3rd_lim)]
        def_3rd = frames[frames[pos_x_col] <= def_3rd_lim]       
 
        if len(start_inds) > 0 and len(end_inds) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one sprint zone for player", player)
                
            start_inds.insert(0, 0)
            end_inds.append(len(sprints)-1)
        
        elif len(sprints) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous sprint zone for player", player)
                
            if ((sprints.iloc[-1].match_time-sprints.iloc[0].match_time)/1000 > 1):
        
                print("Found ONE sustained sprint for player", player)
                no_sprints = 1
        
        # Calculate number of sustained sprints
        for si, ei in zip(start_inds, end_inds):
            
            st = (sprints.iloc[ei].match_time - sprints.iloc[si].match_time)/1000
            
            if st > 1:
                
                no_sprints += 1
        
        accs = frames[frames[acc_col] > 3][[acc_col, 'match_time']].reset_index()
        start_inds_accs = (accs[accs.match_time.diff() > 40].index).to_list()
        end_inds_accs = (accs[accs.match_time.diff() > 40].index-1).to_list()
        decs = frames[frames[acc_col] < -3][[acc_col, 'match_time']].reset_index()
        start_inds_decs = (decs[decs.match_time.diff() > 40].index).to_list()
        end_inds_decs = (decs[decs.match_time.diff() > 40].index-1).to_list()  
        
        if len(start_inds_accs) > 0 and len(end_inds_accs) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one acceleration zone for player", player)
                
            start_inds_accs.insert(0, 0)
            end_inds_accs.append(len(accs)-1)
        
        elif len(accs) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous acceleration zone for player", player)
                
            if ((accs.iloc[-1].match_time-accs.iloc[0].match_time)/1000 > 0.5):
        
                print("Found ONE sustained acceleration for player", player)
                no_accs = 1
   
        if len(start_inds_decs) > 0 and len(end_inds_decs) > 0:
            
            if DEBUG:
                
                print("Found start and end for more than one deceleration zone for player", player)
                
            start_inds_decs.insert(0, 0)
            end_inds_decs.append(len(decs)-1)
        
        elif len(decs) > 0:
            
            if DEBUG:
                
                print("Found ONE continuous deceleration zone for player", player)
                
            if ((decs.iloc[-1].match_time-decs.iloc[0].match_time)/1000 > 0.5):
        
                print("Found ONE sustained deceleration for player", player)
                no_decs = 1
   
        # Calculate number of sustained accelerations.
        for si, ei in zip(start_inds_accs, end_inds_accs):
            
            sa = (accs.iloc[ei].match_time - accs.iloc[si].match_time)/1000
            
            if sa > 0.5:
                
                no_accs += 1
   
         # Calculate number of sustained decelerations.
        if DEBUG:
            
            print("Calculating number of decelerations...")
        
        for si, ei in zip(start_inds_decs, end_inds_decs):
            
            sd = (decs.iloc[ei].match_time - decs.iloc[si].match_time)/1000
            
            if sd > 0.5:
                
                no_decs += 1
   
        tot_dist_cov = frames[vel_col].sum()/25/1000 # In km.
        tot_dist_spr = sprints[vel_col].sum()/25/1000 # In km.
        tot_dist_run = runs[vel_col].sum()/25/1000
        tot_dist_jog = jogs[vel_col].sum()/25/1000
        tot_dist_wal = walks[vel_col].sum()/25/1000
        tot_dist_att_3rd = att_3rd[vel_col].sum()/25/1000
        tot_dist_mid_3rd = mid_3rd[vel_col].sum()/25/1000
        tot_dist_def_3rd = def_3rd[vel_col].sum()/25/1000

        vit_data.loc[player]['DistTot'] = tot_dist_cov
        vit_data.loc[player]['DistWalk'] = tot_dist_wal
        vit_data.loc[player]['DistJog'] = tot_dist_jog
        vit_data.loc[player]['DistRun'] = tot_dist_run
        vit_data.loc[player]['DistSprint'] = tot_dist_spr
        vit_data.loc[player]['NoSprints'] = no_sprints
        vit_data.loc[player]['NoAccels'] = no_accs
        vit_data.loc[player]['NoDecs'] = no_decs
        vit_data.loc[player]['DistAtt3rd'] = tot_dist_att_3rd
        vit_data.loc[player]['DistMid3rd'] = tot_dist_mid_3rd
        vit_data.loc[player]['DistDef3rd'] = tot_dist_def_3rd
    
    return vit_data

###############################################################################
### ------------------------ End of Physics Methods ----------------------- ###
###############################################################################        

###############################################################################
### ----------------------- Start of Helper Methods ----------------------- ###
###############################################################################

def scaleFromPitch(pitch_size, x, y):
    """
    Coordinates in tracking data are pre-scaled to fit playing pitch size.
    For unified plotting, we scale them to fit the OPTA scheme.

    Parameters
    ----------
    pitch_size : (int, int)
        Pitch x/y size in meters.
    x : numpy.array(float)
        array of x values.
    y : numpy.array(float)
        array of y values.

    Returns
    -------
    x : numpy.array(float)
        array of x values.
    y : numpy.array(float)
        array of y values.
    """
    
    x += pitch_size[0]/2
    x /= pitch_size[0]
    x *= 100
    
    y += pitch_size[1]/2
    y /= pitch_size[1]
    y *= 100
    
    return x, y

###############################################################################
### ------------------------ End of Helper Methods ------------------------ ###
###############################################################################

###############################################################################
### ------------------------- Start of Viz Methods ------------------------ ###
###############################################################################


HAM_COL = [("#008200", "#FFFFFF"), ("#FFF200", "#000000"), ("#FFFFFF", "#000000")]
IFE_COL = [("#FFDE00", "#000000"), ("#FF0000", "#000000")]
ORE_COL = [("#FFFFFF", "#000000"), ("#000000", "#FFFFFF")]
MFF_COL = [("#74C2F8", "#FFFFFF"), ("#000060", "#86C7FF"), ("#890000", "#82C1FF")]


class Theme:
    
    homeCol = None
    homeECol = None
    homeTxtCol = None
    awayCol = None
    awayECol = None
    awaTxtCol = None
    bgCol = None
    lnCol = None
    txtCol = 'black'
    view = 'full'
    orientation = 'horizontal'
    inversePlayDir = False
    padLeft = 0
    padRight = 0
    padTop = 0
    padBottom = 0
    figSize = (12, 7.8)
    
    def __init__(self, home, away):
        
        self.homeCol = home[0][0]
        self.homeECol = home[0][1]
        self.homeTxtCol = home[0][1]
        self.awayCol = away[0][0]
        self.awayECol = away[0][1]
        self.awayTxtCol = away[0][1]
    
    
class LightTheme(Theme):
    
    homeCol = 'red'
    homeECol = 'darkred'
    awayCol = 'blue'
    awayECol = 'darkblue'
    bgCol = 'white'
    lnCol = 'grey'
    txtCol = 'black'


class DarkTheme(Theme):
    
    homeCol = 'red'
    homeECol = 'darkred'
    awayCol = 'blue'
    awayECol = 'darkblue'
    # bgCol = '#19232D'
    bgCol = '#313332'
    lnCol = 'lightgrey'
    txtCol = 'white'


class FreezeFrameFeatures:
    
    # Qualifiers.
    voronoi = False
    annotations = False
    ball = False
    velocities = False
    VAR = False
    playerTracks = False
    pitchControl = False
    
    # Data.
    voronoiDataHome = None
    voronoiDataAway = None
    annotationData = None
    ballData = None
    velocitiesHome = None
    velocitiesAway = None
    VARData = None
    playerTrackData = None
    pitchControlData = None
    
    
    def __init__(self, default, voronoi=None, annotations=None, ball=None, velocities=None, VAR=None, playerTracks=None):
        
        assert(type(default) == bool)
        
        if voronoi is None:
            
            self.voronoi = default
            
        else:
            
            self.voronoi = voronoi

        if annotations is None:
            
            self.annotations = default
            
        else:
            
            self.annotations = annotations

        if ball is None:
            
            self.ball = default
            
        else:
            
            self.ball = ball

        if velocities is None:
            
            self.velocities = default
            
        else:
            
            self.velocities = velocities

        if VAR is None:
            
            self.VAR = default
            
        else:
            
            self.VAR = VAR

        if playerTracks is None:
            
            self.playerTracks = default
            
        else:
            
            self.playerTracks = playerTracks
            
    

def gameTimeFormatter(x, pos):
    """
    Helper function to get sensible x axis values for gametime values.
    """
    secs = x//1000
    mins = secs//60
    secs = secs-mins*60
    
    return  "%d:%02d" % (mins, secs)
    

def makingMovies(path, frames, theme, meta_info, features, freq=25):

    animator = animation.writers['ffmpeg']
    meta = dict(title='Tracking Data', artist='MatPlotLib', comment='Assignment 2')
    writer = animator(fps=freq, metadata=meta)
    pitch = mpl_pitch.Pitch(figsize=(16,10.4),
                            pitch_type='wyscout',
                            pitch_color=theme.bgCol,
                            line_color=theme.lnCol)
    fig, ax = pitch.draw()
    fig.patch.set_facecolor(theme.bgCol)
    
    with writer.saving(fig, path, 100):
        
        for i in range(len(frames)):
            
            fig, ax, pitch, trash = plotFreezeFrame(frames, i, meta_info, theme, features, (fig, ax), pitch)
            writer.grab_frame()
            
            for item in trash:
                item.remove()
                
            if DEBUG:
                
                print("Done with frame %d of %d..." % (i, len(frames)))
                

def plotFreezeFrame(frames, iloc, meta_info, theme, features, figax=None, pitch=None):
    
    def rotateCoordinates(x, y, theme):
        
        if theme.orientation == 'horizontal' and theme.inversePlayDir:
            
            x *= -1
            
        elif theme.orientation == 'vertical':
            
            if theme.inversePlayDir:
                
                tmp = x
                x = y
                y = tmp*-1
                
            else:
                
                tmp = x
                x = y
                y = tmp
        
        return x, y
    
    
    if figax is None:
        
        pitch = mpl_pitch.Pitch(figsize=theme.figSize, # 10, 6.5 good for 4x4 panels
                                pitch_type='wyscout',
                                pitch_color=theme.bgCol,
                                line_color=theme.lnCol,
                                line_zorder=1.2,
                                view=theme.view,
                                orientation=theme.orientation,
                                pad_right=theme.padRight,
                                pad_left=theme.padLeft,
                                pad_top=theme.padTop,
                                pad_bottom=theme.padBottom)
        fig, ax = pitch.draw()
        fig.patch.set_facecolor(theme.bgCol)
        
    else:
        
        fig, ax = figax

    frame = frames.iloc[iloc]    
    ho = meta_info.homeTeamName
    aw = meta_info.awayTeamName
    ho_x = []
    ho_y = []
    ho_x_tracks = []
    ho_y_tracks = []
    aw_x = []
    aw_y = []
    aw_x_tracks = []
    aw_y_tracks = []
    ho_vx = []
    ho_vy = []
    aw_vx = []
    aw_vy = []
    anns = []
    hids = meta_info.homeTeamLineup.jersey_number.values
    aids = meta_info.awayTeamLineup.jersey_number.values
    secs = frame.match_time//1000
    mins = secs//60
    secs = secs-mins*60
    track_length = features.playerTracks*25
    start_iloc = iloc-track_length
    
    if start_iloc < 0:
        
        start_iloc = 0
    
    if frame.phase ==2:
        
        mins += 45
    
    var_x = []
    var_y = []
    pitch_x, pitch_y = meta_info.pitchSize
        
    for hid in hids:
        
        x_col = "home_%d_pos_x" % hid
        y_col = "home_%d_pos_y" % hid
        vx_col = "home_%d_pos_x_vel" % hid
        vy_col = "home_%d_pos_y_vel" % hid
        
        if x_col in frame.keys():
            
            # Player not in play right now, skip.
            if np.isnan(frame[x_col]):
                
                continue
            x = frame[x_col]
            y = frame[y_col]
            ho_x.append(x)
            ho_y.append(y)
            ho_vx.append(frame[vx_col])
            ho_vy.append(frame[vy_col])
            anns.append([True, str(hid), (x, y)])
            
            if track_length:
                
                ho_x_track = frames.iloc[start_iloc:iloc][x_col].values.copy()
                ho_y_track = frames.iloc[start_iloc:iloc][y_col].values.copy()
                ho_x_tracks.append(ho_x_track)
                ho_y_tracks.append(ho_y_track)
            
            if DEBUG:
                
                if np.isnan(frame[vx_col]):
                    
                    print("Found vx NaN values for home player %d!" % hid)
                
                if np.isnan(frame[vy_col]):
                    
                    print("Found vy NaN values for home player %d!" % hid)
    
                  
    
    for aid in aids:
    
        x_col = "away_%d_pos_x" % aid
        y_col = "away_%d_pos_y" % aid
        vx_col = "away_%d_pos_x_vel" % aid
        vy_col = "away_%d_pos_y_vel" % aid
        
        if x_col in frame.keys():
            
            # Player not in play right now, skip.
            if np.isnan(frame[x_col]):
                
                continue
            
            x = frame[x_col]
            y = frame[y_col]
            aw_x.append(x)
            aw_y.append(y)
            aw_vx.append(frame[vx_col])
            aw_vy.append(frame[vy_col])
            anns.append([False, str(aid), (x, y)])
            
            if track_length:
                
                aw_x_track = frames.iloc[start_iloc:iloc][x_col].values.copy()
                aw_y_track = frames.iloc[start_iloc:iloc][y_col].values.copy()
                aw_x_tracks.append(aw_x_track)
                aw_y_tracks.append(aw_y_track)
    
            if DEBUG:
                
                if np.isnan(frame[vx_col]):
                    
                    print("Found vx NaN values for away player %d!" % aid)
                
                if np.isnan(frame[vy_col]):
                    
                    print("Found vy NaN values for away player %d!" % aid)
    
    ho_x = np.asarray(ho_x)
    ho_y = np.asarray(ho_y)
    ho_vx = np.asarray(ho_vx)
    ho_vy = np.asarray(ho_vy)
    aw_x = np.asarray(aw_x)
    aw_y = np.asarray(aw_y)
    aw_vx = np.asarray(aw_vx)
    aw_vy = np.asarray(aw_vy)
    var_x = np.asarray(var_x)
    var_y = np.asarray(var_y)
    
    if features.ball:
    
        ba_x, ba_y = frame.ball_pos_x, frame.ball_pos_y
        ba_x, ba_y = scaleFromPitch((pitch_x, pitch_y), ba_x, ba_y)
        ba_x, ba_y = rotateCoordinates(ba_x, ba_y, theme)
        features.ballData = [ba_x, ba_y]
        
    ho_x, ho_y = scaleFromPitch((pitch_x, pitch_y), ho_x, ho_y)
    aw_x, aw_y = scaleFromPitch((pitch_x, pitch_y), aw_x, aw_y)
    ho_x, ho_y = rotateCoordinates(ho_x, ho_y, theme)
    aw_x, aw_y = rotateCoordinates(aw_x, aw_y, theme)
            
    if features.velocities:
        
        ho_vx, ho_vy = rotateCoordinates(ho_vx, ho_vy*-1, theme)
        aw_vx, aw_vy = rotateCoordinates(aw_vx, aw_vy*-1, theme)
        features.velocitiesHome = [ho_vx, ho_vy]
        features.velocitiesAway = [aw_vx, aw_vy]
    
    if features.annotations:
        
        for ann in anns:
            
            x, y = ann[2]
            x, y = scaleFromPitch((pitch_x, pitch_y), x, y)
            x, y = rotateCoordinates(x, y, theme)
            ann[2] = (x, y)
        
        features.annotationData = anns
    
    # VAR.
    # Sort players by x value, pick four as defining the offside line.
    # (Assuming a back four as we're in Sweden, could be defined, whatever...)
    # TODO: Remove players that are not in own half.
    
    if features.VAR:
        
        # Edge case where ball is exactly in the middle of the pitch defined
        # as being on the right side...
        ball_left = (frame.ball_pos_x <= 0)
        
        # Find home keeper half.
        home_keeper = "home_%d_pos_x" % meta_info.homeTeamLineup[meta_info.homeTeamLineup.position == '1'].jersey_number.values[0]
        home_keeper_left = (frame[home_keeper] <= 0)
        
        if ball_left and home_keeper_left:
            
            # Away team attacking right to left.
            var_ind = np.argsort(ho_x)
            var_x = [ho_x[var_ind[1]], ho_x[var_ind[1]], ho_x[var_ind[4]], ho_x[var_ind[4]]]
            var_y = [0, 100, 100, 0]
        
        elif ball_left and (not home_keeper_left):
            
            # Home team attacking right to left.
            var_ind = np.argsort(aw_x)
            var_x = [aw_x[var_ind[1]], aw_x[var_ind[1]], aw_x[var_ind[4]], aw_x[var_ind[4]]]
            var_y = [0, 100, 100, 0]
            
        elif (not ball_left) and (not home_keeper_left):
            
            # Away team attacking left to right.
            var_ind = np.argsort(ho_x)
            var_x = [ho_x[var_ind[-2]], ho_x[var_ind[-2]], ho_x[var_ind[-5]], ho_x[var_ind[-5]]]
            var_y = [0, 100, 100, 0]
            
        else:
            
            # Home team attacking left to right.
            var_ind = np.argsort(aw_x)
            var_x = [aw_x[var_ind[-2]], aw_x[var_ind[-2]], aw_x[var_ind[-5]], aw_x[var_ind[-5]]]
            var_y = [0, 100, 100, 0]
            
        var_x, var_y = rotateCoordinates(var_x, var_y, theme)
        features.VARData = [var_x, var_y]
        
    if features.voronoi:
        
        x = np.concatenate((ho_x, aw_x))
        y = np.concatenate((ho_y, aw_y))
        t = np.concatenate([np.ones(len(ho_x)), np.zeros(len(aw_x))])
        ho_vor, aw_vor = pitch.voronoi(x, y, t)
        features.voronoiDataHome = ho_vor
        features.voronoiDataAway = aw_vor
    
    # Convert player tracks to pitch coordinates.
    if features.playerTracks:
        
        for ho_x_track, ho_y_track in zip(ho_x_tracks, ho_y_tracks):
            
            ho_x_track, ho_y_track = scaleFromPitch((pitch_x, pitch_y), ho_x_track, ho_y_track)
        
        for aw_x_track, aw_y_track in zip(aw_x_tracks, aw_y_tracks):
            
            aw_x_track, aw_y_track = scaleFromPitch((pitch_x, pitch_y), aw_x_track, aw_y_track)
        
        ho_x_tracks, ho_y_tracks = rotateCoordinates(ho_x_tracks, ho_y_tracks, theme)
        aw_x_tracks, aw_y_tracks = rotateCoordinates(aw_x_tracks, aw_y_tracks, theme)
        features.playerTrackData = ((ho_x_tracks, ho_y_tracks), (aw_x_tracks, aw_y_tracks))
    
    if features.pitchControl:
        
        pcm = PitchControlModel(50, meta_info.pitchSize)
        poss_team, pc_surface = pcm.getPitchControlForFrame(frame)
        x_grid, y_grid = pcm.getGridDims()
        x_grid, y_grid = scaleFromPitch((pitch_x, pitch_y), x_grid, y_grid)
        features.pitchControlData = [poss_team, x_grid, y_grid, pc_surface]
    
    trash = plotFreezeFrameEvents((ax, pitch), theme, [ho_x, ho_y], [aw_x, aw_y], features)
    
    title_str = "{} - {}\n".format(ho, aw)
    title_str += "%d:%02d" % (mins, secs)
    ax.set_title(title_str, fontsize=20, color=theme.txtCol)
    
    return fig, ax, pitch, trash


def plotFreezeFrameEvents(ax_pitch, theme, home, away, features):
    """"
    Convenience method for plotting.
    Returns all objects for further destruction when animating.
    """
    ax, pitch = ax_pitch
    trash_bin = []
    ls = 'None'
    m = 'o'
    ms = 25
    ms_ho_co = theme.homeCol
    ms_ho_eco = theme.homeECol
    ms_aw_co = theme.awayCol
    ms_aw_eco = theme.awayECol
    ms_ho_txt_co = theme.homeTxtCol
    ms_aw_txt_co = theme.awayTxtCol
    ho_cmap = mpl_colors.LinearSegmentedColormap.from_list("ho_cmap", [(0, ms_ho_co), (1, ms_ho_eco)])
    aw_cmap = mpl_colors.LinearSegmentedColormap.from_list("aw_cmap", [(0, ms_aw_co), (1, ms_aw_eco)])
    
    trash, = pitch.plot(home[0], home[1], marker=m, markersize=ms, markerfacecolor=ms_ho_co, markeredgecolor=ms_ho_eco, linestyle=ls, zorder=3, ax=ax)
    trash_bin.append(trash)
    trash, = pitch.plot(away[0], away[1], marker=m, markersize=ms, markerfacecolor=ms_aw_co, markeredgecolor=ms_aw_eco, linestyle=ls, zorder=2, ax=ax)
    trash_bin.append(trash)
    
    if features.velocities:
        
        home_vel = features.velocitiesHome
        away_vel = features.velocitiesAway
        trash = ax.quiver(home[0], home[1], home_vel[0], home_vel[1], color=ms_ho_eco, scale_units='inches', scale=7.5, width=0.006, headlength=5, headwidth=3, zorder=2.9)
        trash_bin.append(trash)
        trash = ax.quiver(away[0], away[1], away_vel[0], away_vel[1], color=ms_aw_eco, scale_units='inches', scale=7.5, width=0.006, headlength=5, headwidth=3, zorder=1.9)
        trash_bin.append(trash)

    if features.ball:
        
        ball = features.ballData
        trash = pitch.scatter(ball[0], ball[1], marker='football', s=250, zorder=4, ax=ax)
        trash_bin.append(trash[0])
        trash_bin.append(trash[1])

    if features.annotations:
        
        for ho, txt, xy in features.annotationData:
            
            if ho:
                
                trash = ax.annotate(txt, xy, va='center', ha='center', c=ms_ho_txt_co, fontsize=12, zorder=3.1)
                
            else:
                
                trash = ax.annotate(txt, xy, va='center', ha='center', c=ms_aw_txt_co, fontsize=12, zorder=2.1)
                
            trash_bin.append(trash)
           
    if features.VAR:
        
        var = features.VARData
        alpha = 0.33
        
        if features.voronoi:
            
            alpha = 1
        
        trash, = ax.fill(var[0], var[1], color='red', alpha=alpha, zorder=0.9)
        trash_bin.append(trash)
    
    # Voronoi.
    if features.voronoi:
        
        t1 = pitch.polygon(features.voronoiDataHome, ax=ax, fc=ms_ho_co, ec=ms_ho_eco, lw=2, alpha=0.66)
        t2 = pitch.polygon(features.voronoiDataAway, ax=ax, fc=ms_aw_co, ec=ms_aw_eco, lw=2, alpha=0.66)
        trash_bin.append(t1)
        trash_bin.append(t2)
        
    if features.playerTracks:
        
        ho_x_tracks, ho_y_tracks = features.playerTrackData[0]
        aw_x_tracks, aw_y_tracks = features.playerTrackData[1]
        
        for ho_x_track, ho_y_track in zip(ho_x_tracks, ho_y_tracks):
            
            track_len = len(ho_x_track)
            c = np.arange(track_len)
            norm = mpl_colors.Normalize(vmin=0, vmax=track_len)
            trash = pitch.scatter(ho_x_track, ho_y_track, s=2, c=c, cmap=ho_cmap, norm=norm, zorder=1.9, ax=ax)
            trash_bin.append(trash)

        for aw_x_track, aw_y_track in zip(aw_x_tracks, aw_y_tracks):
            
            track_len = len(aw_x_track)
            c = np.arange(track_len)
            trash = pitch.scatter(aw_x_track, aw_y_track, s=2, c=c, cmap=aw_cmap, norm=norm, zorder=1.9, ax=ax)
            trash_bin.append(trash)    
    
    if features.pitchControl:
        
        poss_team, x_grid, y_grid, pc_surface = features.pitchControlData
        
        if poss_team == 'away':
            
            pc_cmap = mpl_colors.LinearSegmentedColormap.from_list("pc_cmap", [ms_ho_co, 'white', ms_aw_co])

        else:
            
            pc_cmap = mpl_colors.LinearSegmentedColormap.from_list("pc_cmap", [ms_aw_co, 'white', ms_ho_co])
        
        # trash = ax.imshow(np.flipud(pc_surface), extent=(np.amin(x_grid), np.amax(x_grid), np.amin(y_grid), np.amax(y_grid)), interpolation='hanning', vmin=0.0, vmax=1.0, cmap=pc_cmap, alpha=0.66, zorder=1.5)
        trash = ax.imshow(pc_surface, origin='lower', extent=(np.amin(x_grid), np.amax(x_grid), np.amin(y_grid), np.amax(y_grid)), interpolation='hanning', vmin=0.0, vmax=1.0, cmap=pc_cmap, zorder=1.1)
        trash_bin.append(trash)
        ax.set_aspect(69/104)
        cb = ax.figure.colorbar(trash, ticks=[0,0.25,0.5,0.75,1])
        # cb.set_label('Pitch Control', color=theme.txtCol, size=12, weight='bold')
        cb.ax.set_yticklabels(['100% Away', '75% Away', '50/50', '75% Home', '100 %Home'], color=theme.txtCol)
        cb.ax.yaxis.set_tick_params(color=theme.lnCol)
        cb.outline.set_edgecolor(theme.lnCol)
        trash_bin.append(cb)
        
    return trash_bin


def poorMans1DVoronoi(frames, meta_info, theme, player):
    """
    Part 3 of the assignment.
    Finding closest opponent and teammate.
    Plotting as function of time.
    """
    ms_ho_co = theme.homeCol
    ms_ho_eco = theme.homeECol
    ms_aw_co = theme.awayCol
    ms_aw_eco = theme.awayECol
    ho_cmap = mpl_colors.LinearSegmentedColormap.from_list("ho_cmap", [(0, ms_ho_co), (1, ms_ho_eco)])
    aw_cmap = mpl_colors.LinearSegmentedColormap.from_list("aw_cmap", [(0, ms_aw_co), (1, ms_aw_eco)])
    fig, ax = plt.subplots(figsize=(10,6))
    
    vel_col = 'home_%d_pos_vel' % player
    acc_col = 'home_%d_pos_acc' % player
    
    time_frame = frames.match_time.values
    hids = meta_info.homeTeamLineup.jersey_number.values
    ho_names = meta_info.homeTeamLineup.name.values
    
    aids = meta_info.awayTeamLineup.jersey_number.values
    aw_names = meta_info.awayTeamLineup.name.values

    t0 = time_frame[0]
    t1 = time_frame[-1]
    oppo_players = {}
    team_mates = {}
    
    label = ho_names[np.where(hids == player)][0].split(' ')[-1] + ' (%d)' % player
    
    # Iterate and find closest teammate and opposition player.
    for i in range(len(frames)):
        
        frame = frames.iloc[i]
        
        home = pd.json_normalize(frame.home_team)
        away = pd.json_normalize(frame.away_team)
                
        player_pos_index = home.loc[home.jersey_number == player].index[0]
        player_pos = home.iloc[player_pos_index].position
        oppo_posses = np.asarray(away.position.to_list())
        home_posses = home.position.to_list()
        player_pos_id = home_posses.index(player_pos)
        home_posses.remove(player_pos)
        home_posses = np.asarray(home_posses)
        player_pos = np.asarray(player_pos)
        tree = cKDTree(oppo_posses)
        oppo_dist, oppo_id = tree.query(player_pos)
        oppo_jersey = away.iloc[oppo_id].jersey_number
        tree = cKDTree(home_posses)
        team_dist, team_id = tree.query(player_pos)
        
        # Add 1 to the index if it's equal or larger than the index we removed.
        if team_id >= player_pos_id:
            
            team_id +=1
            
        team_jersey = home.iloc[team_id].jersey_number
        
        if oppo_jersey in oppo_players:
            
            oppo_players[oppo_jersey][0].append(frames.iloc[i].match_time)
            oppo_players[oppo_jersey][1].append(oppo_dist)
    
        else:
            
            oppo_players[oppo_jersey] = ([frames.iloc[i].match_time], [oppo_dist])
            
        if team_jersey in team_mates:
            
            team_mates[team_jersey][0].append(frames.iloc[i].match_time)
            team_mates[team_jersey][1].append(team_dist)
    
        else:
            
            team_mates[team_jersey] = ([frames.iloc[i].match_time], [team_dist])
            
    ho_co_map = ho_cmap(np.linspace(0, 1, len(team_mates)+1))
    aw_co_map = aw_cmap(np.linspace(0, 1, len(oppo_players)+1))
    ho_co_cycler = cycler.cycler('color', ho_co_map)
    aw_co_cycler = cycler.cycler('color', aw_co_map)
    
    # Set team mates first.
    ax.set_prop_cycle(ho_co_cycler)
    zeros = np.zeros(len(time_frame))
    line, = ax.plot(time_frame, zeros, lw=5, zorder=2, label=label)
    ax.fill_between(time_frame, frames[vel_col], zeros, edgecolor=line.get_color(), facecolor='None', hatch='///', alpha=0.66, zorder=1.8, label='Speed')
    ax.fill_between(time_frame, frames[acc_col], zeros, edgecolor=ms_ho_eco, facecolor='None', hatch='\\\\\\', alpha=0.66, zorder=1.9, label='Acceleration')
    
    for team_mate in team_mates.keys():
        
        x, y = team_mates[team_mate]
        label = ho_names[np.where(hids == team_mate)][0].split(' ')[-1] + ' (%d)' % team_mate
        scat = ax.scatter(x, y, zorder=2, label=label, ec=ms_ho_eco, s=200, lw=0.5)
        ax.plot([x[-1], x[-1]], [y[-1]-2, y[-1]+2], lw=2, color=scat.get_facecolor()[0], zorder=1.9)
        print("Mate:", x[-1])
    
    # Now oppositions players.
    ax.set_prop_cycle(aw_co_cycler)
    
    for oppo_player in oppo_players.keys():
        
        x, y = oppo_players[oppo_player]
        label = aw_names[np.where(aids == oppo_player)][0].split(' ')[-1] + ' (%d)' % oppo_player
        scat = ax.scatter(x, y, zorder=2, label=label, ec=ms_aw_eco, s=200, lw=0.5)
        ax.plot([x[-1], x[-1]], [y[-1]-2, y[-1]+2], lw=2, color=scat.get_facecolor()[0], zorder=1.9)
        print("Oppo:", x[-1])
    
    
    ax.axvline(1805071, ymin=0.1, ymax=0.9, c=theme.lnCol, label='Assist (Tankovic)', zorder=1, linestyle='--')
    ax.axvline(1807471, ymin=0.1, ymax=0.9, c=theme.lnCol, label='Shot (Khalili)', zorder=1, linestyle='-')
    
    
    ax.plot(1801590, 6.5, marker='o', ms=25, color=theme.lnCol, zorder=3)
    ax.plot(1803471, 10, marker='o', ms=25, color=theme.lnCol, zorder=3)
    ax.plot(1805071, 14.5, marker='o', ms=25, color=theme.lnCol, zorder=3)
    ax.plot(1807471, 14.5, marker='o', ms=25, color=theme.lnCol, zorder=3)
    
    ax.annotate("1", (1801590, 6.5), va='center', ha='center', c=theme.bgCol, fontsize=12, zorder=3.1)
    ax.annotate("2", (1803471, 10), va='center', ha='center', c=theme.bgCol, fontsize=12, zorder=3.1)
    ax.annotate("3", (1805071, 14.5), va='center', ha='center', c=theme.bgCol, fontsize=12, zorder=3.1)
    ax.annotate("4", (1807471, 14.5), va='center', ha='center', c=theme.bgCol, fontsize=12, zorder=3.1)
    
    ax.set_facecolor(theme.bgCol)
    ax.set_xlim([t0, t1])
    ax.spines['bottom'].set_color(theme.lnCol)
    ax.spines['top'].set_color(theme.lnCol)
    ax.spines['left'].set_color(theme.lnCol)
    ax.spines['right'].set_color(theme.lnCol)
    ax.set_ylabel('Distance to nearest player [m]', weight='bold')
    ax.xaxis.label.set_color(theme.txtCol)
    ax.yaxis.label.set_color(theme.txtCol)
    ax.tick_params(axis='x', colors=theme.lnCol)
    ax.tick_params(axis='y', colors=theme.lnCol)
    ax.xaxis.set_major_formatter(gameTimeFormatter)
    ax.set_xlabel('Match time [mm:ss]', weight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)
    fig.patch.set_facecolor(theme.bgCol)
    fig.tight_layout()


def vitalityViz(frames, meta_info, theme, freeze_frame=None, errors=None):
    ms_ho_co = theme.homeCol
    ms_ho_eco = theme.homeECol
    ms_aw_co = theme.awayCol
    ms_aw_eco = theme.awayECol
    ho_cmap = mpl_colors.LinearSegmentedColormap.from_list("ho_cmap", [(0, ms_ho_co), (1, ms_ho_eco)])
    aw_cmap = mpl_colors.LinearSegmentedColormap.from_list("aw_cmap", [(0, ms_aw_co), (1, ms_aw_eco)])
    ho_co_map = ho_cmap(np.linspace(0, 1, 6))
    aw_co_map = aw_cmap(np.linspace(0, 1, 6))
    
    ho_co_cycler = cycler.cycler('color', ho_co_map)
    aw_co_cycler = cycler.cycler('color', aw_co_map)
    fig, axes = plt.subplots(3,1, figsize=(8,10), sharex=True, constrained_layout=True)
    
    goal = -meta_info.pitchSize[0]/2
    axes[0].set_ylabel('Distance from goal [m]', weight='bold', size=16)
    axes[0].set_prop_cycle(ho_co_cycler)
    axes[1].set_ylabel('Speed [m/s]', weight='bold')
    axes[1].set_prop_cycle(ho_co_cycler)
    axes[2].set_ylabel('Acceleration [m/s^2]', weight='bold', size=16)
    axes[2].set_xlabel('Match time [mm:ss]', weight='bold', size=16)
    axes[2].set_prop_cycle(ho_co_cycler)
    time_frame = frames.match_time.values
    hids = meta_info.homeTeamLineup.jersey_number.values
    ho_names = meta_info.homeTeamLineup.name.values
    ho_positions = meta_info.homeTeamLineup.position.values
    
    aids = meta_info.awayTeamLineup.jersey_number.values
    aw_names = meta_info.awayTeamLineup.name.values
    aw_positions = meta_info.awayTeamLineup.position.values
    
    
    t0 = time_frame[0]
    t1 = time_frame[-1]
    fft = None
    
    if freeze_frame is not None:
    
        fft = time_frame[freeze_frame]
    
    players = []
    player_names = []
    
    for hid, name, pos in zip(hids, ho_names, ho_positions):
        
        # This is an evil, time-constrained hack to just look at the front four.
        if int(pos) not in [7, 9, 10, 11]:
            
            continue
        
        px_col = "home_%d_pos_x" % hid
        py_col = "home_%d_pos_y" % hid
        v_col = "home_%d_pos_vel" % hid
        a_col = "home_%d_pos_acc" % hid
        player = "home_%d" % hid
        
        # No data for this player, skip.
        if not v_col in frames.columns:
            
            continue
        
        # No valid data for this player, skip.
        if frames[v_col].isna().sum() == frames[v_col].isna().count():
            
            continue
        
        players.append(player)
        player_names.append(name.split(' ')[-1])
        label = name.split(' ')[-1] + " (%d)" % hid
        dist_to_goal = ((goal-frames[px_col])**2 + (0-frames[py_col])**2)**0.5
        axes[0].plot(time_frame, dist_to_goal, lw=3, zorder=2, label=label)
        axes[1].plot(time_frame, frames[v_col], lw=3, zorder=2, label=label)
        axes[2].plot(time_frame, frames[a_col], lw=3, zorder=2, label=label)

    
    axes[0].set_prop_cycle(aw_co_cycler)
    axes[1].set_prop_cycle(aw_co_cycler)
    axes[2].set_prop_cycle(aw_co_cycler)
    
    oppo_av_d2g = None
    oppo_av_vel = None
    oppo_av_acc = None
    
    for aid, name, pos in zip(aids, aw_names, aw_positions):
        
        # This is an evil, time-constrained hack to just look at the back four.
        if int(pos) not in [2, 3, 4, 5]:
            
            continue
        
        px_col = "away_%d_pos_x" % aid
        py_col = "away_%d_pos_y" % aid
        v_col = "away_%d_pos_vel" % aid
        a_col = "away_%d_pos_acc" % aid
        player = "away_%d" % aid
        
        # No data for this player, skip.
        if not v_col in frames.columns:
            
            continue
        
        # No valid data for this player, skip.
        if frames[v_col].isna().sum() == frames[v_col].isna().count():
            
            continue
        
        
        label = name.split(' ')[-1] + " (%d)" % aid
        dist_to_goal = ((goal-frames[px_col])**2 + (0-frames[py_col])**2)**0.5
        
        if oppo_av_d2g is None:
            
            oppo_av_d2g = dist_to_goal
            oppo_av_vel = frames[v_col].values
            oppo_av_acc = frames[a_col].values
            
        else:
            
            oppo_av_d2g += dist_to_goal
            oppo_av_d2g /= 2
            oppo_av_vel += frames[v_col].values
            oppo_av_vel /= 2
            oppo_av_acc += frames[a_col].values
            oppo_av_acc /= 2
            
    axes[0].plot(time_frame, oppo_av_d2g, ls='--', zorder=1.5, alpha=0.66, label='Back 4 avg.')
    
    axes[1].plot(time_frame, oppo_av_vel, ls='--', alpha=0.66, zorder=1.5, label='Back 4 avg.')
    axes[2].plot(time_frame, oppo_av_acc, ls='--', alpha=0.66, zorder=1.5, label='Back 4 avg.')
    
    if freeze_frame is not None:
        
        axes[0].axvline(fft, ls='--', c=theme.lnCol, label='Freeze frame', zorder=1.1)
        axes[1].axvline(fft, ls='--', c=theme.lnCol, label='Freeze frame', zorder=1.1)
        axes[2].axvline(fft, ls='--', c=theme.lnCol, label='Freeze frame', zorder=1.1)

    if errors is not None:
        
        for ets in errors:
            
            et0, et1 = ets
            axes[0].fill([et0, et1, et1, et0], [0, 0, 100, 100], color='red', alpha=0.33, zorder=1.1)
            axes[1].fill([et0, et1, et1, et0], [0, 0, 10, 10], color='red', alpha=0.33, zorder=1.1)
            axes[2].fill([et0, et1, et1, et0], [-7.5, -7.5, 7.5, 7.5], color='red', alpha=0.33, zorder=1.1)
            
    # Limits and backgrounds.
    axes[0].set_ylim([0, 50])
    axes[1].set_ylim([0, 10])
    axes[2].set_ylim([-7.5, 7.5])
    axes[0].set_xlim([t0, t1])
    axes[1].set_xlim([t0, t1])
    axes[2].set_xlim([t0, t1])
    axes[0].fill([t0, t1, t1, t0], [66.01, 66.01, 100, 100], color='white', alpha=0.75, zorder=1)
    axes[0].fill([t0, t1, t1, t0], [33.01, 33.01, 66, 66], color='white', alpha=0.5, zorder=1, label='Central 3rd')
    axes[0].fill([t0, t1, t1, t0], [0, 0, 33, 33], color='white', alpha=0.25, zorder=1, label='Attacking 3rd')
    axes[1].fill([t0, t1, t1, t0], [7, 7, 10, 10], color='white', alpha=0.6, zorder=1, label='Sprinting')
    axes[1].fill([t0, t1, t1, t0], [4, 4, 7, 7], color='white', alpha=0.45, zorder=1, label='Running')
    axes[1].fill([t0, t1, t1, t0], [2, 2, 4, 4], color='white', alpha=0.3, zorder=1, label='Jogging')
    axes[1].fill([t0, t1, t1, t0], [0, 0, 2, 2], color='white', alpha=0.15, zorder=1, label='Walking')
    axes[2].fill([t0, t1, t1, t0], [4.1, 4.1, 7.5, 7.5], color='white', alpha=0.6, zorder=1, label='Very high')
    axes[2].fill([t0, t1, t1, t0], [3.1, 3.1, 4, 4], color='white', alpha=0.45, zorder=1, label='High')
    axes[2].fill([t0, t1, t1, t0], [1.51, 1.51, 3, 3], color='white', alpha=0.30, zorder=1, label='Moderate')
    axes[2].fill([t0, t1, t1, t0], [-1.5, -1.5, 1.5, 1.5], color='white', alpha=0.15, zorder=1, label='Low')
    axes[2].fill([t0, t1, t1, t0], [-4.1, -4.1, -7.5, -7.5], color='white', alpha=0.6, zorder=1)
    axes[2].fill([t0, t1, t1, t0], [-3.1, -3.1, -4, -4], color='white', alpha=0.45, zorder=1)
    axes[2].fill([t0, t1, t1, t0], [-1.51, -1.51, -3, -3], color='white', alpha=0.30, zorder=1)
    
    for ax in axes:
        
        # ax.grid(which='major', axis='x', linestyle='--', zorder=0.9)
        ax.set_facecolor(theme.bgCol)
        ax.spines['bottom'].set_color(theme.lnCol)
        ax.spines['top'].set_color(theme.lnCol)
        ax.spines['left'].set_color(theme.lnCol)
        ax.spines['right'].set_color(theme.lnCol)
        ax.xaxis.label.set_color(theme.txtCol)
        ax.yaxis.label.set_color(theme.txtCol)
        ax.tick_params(axis='x', colors=theme.lnCol)
        ax.tick_params(axis='y', colors=theme.lnCol)
        ax.xaxis.set_major_formatter(gameTimeFormatter)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
        plt.setp(legend.get_texts(), color=theme.txtCol, fontsize=16)
        plt.setp(legend.get_patches(), ec=theme.txtCol)
        plt.setp(ax.get_xticklabels(), fontsize=16, color=theme.txtCol)
        plt.setp(ax.get_yticklabels(), fontsize=16, color=theme.txtCol)
    
    axes[0].set_title("%s - %s" % (meta_info.homeTeamName, meta_info.awayTeamName), color=theme.txtCol, size=18, weight='bold')
    fig.patch.set_facecolor(theme.bgCol)
    

def vitalityVizSummary(vit_data, meta_info, theme):
    
    fig, axes = plt.subplots(3,2, figsize=(16,12), sharex='row')
    ms_ho_co = theme.homeCol
    ms_ho_eco = theme.homeECol
    ms_aw_co = theme.awayCol
    ms_aw_eco = theme.awayECol
    ho_cmap = mpl_colors.LinearSegmentedColormap.from_list("ho_cmap", [(0, ms_ho_eco), (1, ms_ho_co)])
    aw_cmap = mpl_colors.LinearSegmentedColormap.from_list("aw_cmap", [(0, ms_aw_eco), (1, ms_aw_co)])
    
    axes[0][0].set_xlabel('Distance run by pitch thirds [km]', weight='bold')
    axes[1][0].set_xlabel('Distance run by speed [km]', weight='bold')
    axes[2][0].set_xlabel('Sustained high intensity actions [#]', weight='bold')
    axes[0][1].set_xlabel('Distance run by pitch thirds [km]', weight='bold')    
    axes[1][1].set_xlabel('Distance run by speed [km]', weight='bold')
    axes[2][1].set_xlabel('Sustained high intensity actions[#]', weight='bold')
    hids = meta_info.homeTeamLineup.jersey_number.values
    aids = meta_info.awayTeamLineup.jersey_number.values
    ho_names = meta_info.homeTeamLineup.name.values
    aw_names = meta_info.awayTeamLineup.name.values
    ho_players = []
    ho_player_names = []
    aw_players = []
    aw_player_names = []
    
    for hid, ho_name in zip(hids, ho_names):
        
        ho_player = "home_%d" % hid
        
        if vit_data.loc[ho_player].sum() == 0:
            
            continue
        
        ho_players.append(ho_player)
        ho_player_names.append(ho_name.split(' ')[-1])

    for aid, aw_name in zip(aids, aw_names):
        
        aw_player = "away_%d" % aid
        
        if vit_data.loc[aw_player].sum() == 0:
            
            continue
        
        aw_players.append(aw_player)
        aw_player_names.append(aw_name.split(' ')[-1])

    ho_co_map = ho_cmap(np.linspace(0, 1, len(ho_players)))
    aw_co_map = aw_cmap(np.linspace(0, 1, len(aw_players)))
    
    home_vit = vit_data.loc[ho_players]
    away_vit = vit_data.loc[aw_players]
    
    # Bar chart bonanzas.
    bar_height = 0.35
    y = np.arange(len(home_vit))
    
    # Home team.
    axes[0][0].barh(y+bar_height/2, home_vit.DistTot, height=bar_height, color=ho_co_map, zorder=2)
    axes[0][0].barh(y-bar_height/2, home_vit.DistDef3rd, height=bar_height, color='white', alpha=0.25, label='Defensive 3rd', zorder=2)
    left = home_vit.DistDef3rd
    axes[0][0].barh(y-bar_height/2, home_vit.DistMid3rd, height=bar_height, left=left, color='white', alpha=0.50, label='Middle 3rd', zorder=2)
    left += home_vit.DistMid3rd
    axes[0][0].barh(y-bar_height/2, home_vit.DistAtt3rd, height=bar_height, left=left, color='white', alpha=0.75, label='Attacking 3rd', zorder=2)
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels(ho_player_names, weight='bold')
    axes[0][0].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[0][0].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)
    
    axes[1][0].barh(y+bar_height/2, home_vit.DistTot, height=bar_height, color=ho_co_map, zorder=2)
    axes[1][0].barh(y-bar_height/2, home_vit.DistWalk, height=bar_height, color='white', alpha=0.15, label='Walking', zorder=2)
    left = home_vit.DistWalk
    axes[1][0].barh(y-bar_height/2, home_vit.DistJog, height=bar_height, left=left, color='white', alpha=0.3, label='Jogging', zorder=2)
    left += home_vit.DistJog
    axes[1][0].barh(y-bar_height/2, home_vit.DistRun, height=bar_height, left=left, color='white', alpha=0.45, label='Running', zorder=2)
    left += home_vit.DistRun
    axes[1][0].barh(y-bar_height/2, home_vit.DistSprint, height=bar_height, left=left, color='white', alpha=0.6, label='Sprinting', zorder=2)
    axes[1][0].set_yticks(y)
    axes[1][0].set_yticklabels(ho_player_names, weight='bold')
    axes[1][0].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[1][0].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)

    axes[2][0].barh(y+bar_height/2, home_vit.NoAccels+home_vit.NoDecs+home_vit.NoSprints, height=bar_height, color=ho_co_map, zorder=2)
    axes[2][0].barh(y-bar_height/2, home_vit.NoSprints, height=bar_height, color='white', alpha=0.3, label='Sprints', zorder=2)
    axes[2][0].barh(y-bar_height/2, home_vit.NoAccels, left=home_vit.NoSprints, height=bar_height, color='white', alpha=0.45, label='Accelerations', zorder=2)
    axes[2][0].barh(y-bar_height/2, home_vit.NoDecs, left=home_vit.NoSprints+home_vit.NoAccels, height=bar_height, color='white', alpha=0.6, label='Decelerations', zorder=2)
    axes[2][0].set_yticks(y)
    axes[2][0].set_yticklabels(ho_player_names, weight='bold')
    axes[2][0].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[2][0].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)
    
    # Away team.
    axes[0][1].barh(y+bar_height/2, away_vit.DistTot, height=bar_height, color=aw_co_map, zorder=2)
    axes[0][1].barh(y-bar_height/2, away_vit.DistDef3rd, height=bar_height, color='white', alpha=0.25, label='Defensive 3rd', zorder=2)
    left = away_vit.DistDef3rd
    axes[0][1].barh(y-bar_height/2, away_vit.DistMid3rd, height=bar_height, left=left, color='white', alpha=0.50, label='Middle 3rd', zorder=2)
    left += away_vit.DistMid3rd
    axes[0][1].barh(y-bar_height/2, away_vit.DistAtt3rd, height=bar_height, left=left, color='white', alpha=0.75, label='Attacking 3rd', zorder=2)
    axes[0][1].set_yticks(y)
    axes[0][1].set_yticklabels(aw_player_names, weight='bold')
    axes[0][1].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[0][1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)
    
    axes[1][1].barh(y+bar_height/2, away_vit.DistTot, height=bar_height, color=aw_co_map, zorder=2)
    axes[1][1].barh(y-bar_height/2, away_vit.DistWalk, height=bar_height, color='white', alpha=0.15, label='Walking', zorder=2)
    left = away_vit.DistWalk
    axes[1][1].barh(y-bar_height/2, away_vit.DistJog, height=bar_height, left=left, color='white', alpha=0.3, label='Jogging', zorder=2)
    left += away_vit.DistJog
    axes[1][1].barh(y-bar_height/2, away_vit.DistRun, height=bar_height, left=left, color='white', alpha=0.45, label='Running', zorder=2)
    left += away_vit.DistRun
    axes[1][1].barh(y-bar_height/2, away_vit.DistSprint, height=bar_height, left=left, color='white', alpha=0.6, label='Sprinting', zorder=2)
    axes[1][1].set_yticks(y)
    axes[1][1].set_yticklabels(aw_player_names, weight='bold')
    axes[1][1].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[1][1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)

    axes[2][1].barh(y+bar_height/2, away_vit.NoAccels+away_vit.NoDecs+away_vit.NoSprints, height=bar_height, color=aw_co_map, zorder=2)
    axes[2][1].barh(y-bar_height/2, away_vit.NoSprints, height=bar_height, color='white', alpha=0.3, label='Sprints', zorder=2)
    axes[2][1].barh(y-bar_height/2, away_vit.NoAccels, left=away_vit.NoSprints, height=bar_height, color='white', alpha=0.45, label='Accelerations', zorder=2)
    axes[2][1].barh(y-bar_height/2, away_vit.NoDecs, left=away_vit.NoSprints+away_vit.NoAccels, height=bar_height, color='white', alpha=0.6, label='Decelerations', zorder=2)
    axes[2][1].set_yticks(y)
    axes[2][1].set_yticklabels(aw_player_names, weight='bold')
    axes[2][1].tick_params(axis="y", length=0, labelrotation=45)
    legend = axes[2][1].legend(loc='center left', bbox_to_anchor=(1.01, 0.5), facecolor=theme.bgCol, edgecolor='None')
    plt.setp(legend.get_texts(), color=theme.txtCol)
    plt.setp(legend.get_patches(), ec=theme.txtCol)
    
    # Apply theme colors.
    fig.patch.set_facecolor(theme.bgCol)
    
    for ax in axes:
        
        ax[0].set_facecolor(theme.bgCol)
        ax[0].spines['bottom'].set_color(theme.lnCol)
        ax[0].spines['top'].set_color(theme.lnCol)
        ax[0].spines['left'].set_color(theme.lnCol)
        ax[0].spines['right'].set_color(theme.lnCol)
        ax[0].xaxis.label.set_color(theme.txtCol)
        ax[0].yaxis.label.set_color(theme.txtCol)
        ax[0].tick_params(axis='x', colors=theme.lnCol)
        ax[0].tick_params(axis='y', colors=theme.lnCol)
        ax[0].xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].grid(axis='x', zorder=0.5, ls='--')
    
        ax[1].set_facecolor(theme.bgCol)
        ax[1].spines['bottom'].set_color(theme.lnCol)
        ax[1].spines['top'].set_color(theme.lnCol)
        ax[1].spines['left'].set_color(theme.lnCol)
        ax[1].spines['right'].set_color(theme.lnCol)
        ax[1].xaxis.label.set_color(theme.txtCol)
        ax[1].yaxis.label.set_color(theme.txtCol)
        ax[1].tick_params(axis='x', colors=theme.lnCol)
        ax[1].tick_params(axis='y', colors=theme.lnCol)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))
        ax[1].grid(axis='x', zorder=0.5, ls='--')

    fig.suptitle("Head-to-head vitality\n%s - %s (1st half only)" % (meta_info.homeTeamName, meta_info.awayTeamName),
                  color=theme.txtCol, size=16, weight='bold')
    fig.tight_layout()



def vizHAM_IFE(ht, at, movies=False):
    
    theme = DarkTheme(HAM_COL, IFE_COL)
    # theme = LightTheme(HAM_COL, IFE_COL)
    theme.padRight = -77
    features = FreezeFrameFeatures(True)
    features.voronoi = False
    features.pitchControl = False
    features.playerTracks = 5
    
    # Load first match in list.
    print("Loading first half data for HAM-IFE...")
    events1, info, stats1, tracks1 = getSignalityData(ht, at, 1)
    # print("Loading second half data...")
    # events2, info, stats2, tracks2 = getSignalityData(ht, at, 2)
    
    # Get team lineups.
    hids = info.homeTeamLineup.jersey_number.values
    aids = info.awayTeamLineup.jersey_number.values

    # Interpolate ball positions. This is bad, but it is what it is.
    print("Interpolating ball positions...")
    tracks1.ball_pos_x.interpolate(method='linear', inplace=True)
    tracks1.ball_pos_y.interpolate(method='linear', inplace=True)
    tracks1.ball_player.interpolate(method='pad', inplace=True)
    # tracks2.ball_pos_x.interpolate(method='linear', inplace=True)
    # tracks2.ball_pos_y.interpolate(method='linear', inplace=True)
    
    # # Calculate velocities without filterting.
    print("Calculating velocity, speed and acceleration...")
    tracks_raw = calcVelocities(tracks1, hids, aids, filtering=False, interp_nans=True)
    tracks_smooth = calcVelocities(tracks1, hids, aids, filtering=True, interp_nans=True)
    
    # frame = tracks_raw.iloc[45250]
    # balls = plotFreezeFrame(frame, info, light_theme, features)
    # frame = tracks_smooth.iloc[45250]
    # balls = plotFreezeFrame(frame, info, light_theme, features)
    
    # Set piece example of bad raw tracking data:
    frames = tracks_raw[5000:5375]
    vitalityViz(frames, info, theme, None)
    frames = tracks_smooth[5000:5375]
    vitalityViz(frames, info, theme, None)
    

    # # HAM 1-0 IFE
    # frames = tracks_raw[26500:26875]
    
    # if movies:
    
    #     balls = makingMovies("HAM-IFE 1-0 NOFILT.mp4", frames, dark_theme, info, features)
    
    frames = tracks_smooth[26500:26875]
    
    if movies:
        
        print("Making movie of HAM-IFE 1-0...")
        balls = makingMovies("HAM-IFE 1-0.mp4", frames, theme, info, features)
        print("Done!")

    print("Displaying freeze frame just before 1-0 goal.")
    vitalityViz(frames, info, theme, 325)
    features.pitchControl = False
    plotFreezeFrame(frames, 325, info, theme, features)
    plotFreezeFrame(frames, 325, info, theme, features)

    # Plot four freeze frames highlighting Đurđić's run.
    plotFreezeFrame(frames, 260, info, theme, features)
    plotFreezeFrame(frames, 275, info, theme, features)
    plotFreezeFrame(frames, 300, info, theme, features)
    plotFreezeFrame(frames, 325, info, theme, features)

    # # HAM 2-0 IFE
    # frames = tracks_raw[40900:41200]
    # balls = makingMovies("HAM-IFE 2-0 NOFILT.mp4", frames, light_theme, info, features)
    # frames = tracks_smooth[40900:41200]
    # balls = makingMovies("HAM-IFE 2-0.mp4", frames, light_theme, info, features)
    
    # # HAM 3-0 IFE
    # frames = tracks_raw[44750:45250]
    # raw_vit_data = getVitalityData(info, frames)
    # balls = makingMovies("HAM-IFE 3-0 NOFILT.mp4", frames, light_theme, info, features)
    frames = tracks_smooth[44850:45225]
    
    if movies:
        
        print("Making movie of HAM-IFE 3-0...")
        balls = makingMovies("HAM-IFE 3-0.mp4", frames, theme, info, features)
        print("Done!")
    
    vitalityViz(frames, info, theme, 265)
    poorMans1DVoronoi(frames, info, theme, 7)
    print("Displaying freeze frame just before 3-0 goal.")
    plotFreezeFrame(frames, 265, info, theme, features) # Frame 3: Assist 
    features.pitchControl = True
    plotFreezeFrame(frames, 178, info, theme, features) # Frame 1: Dummy run
    plotFreezeFrame(frames, 225, info, theme, features) # Frame 2: Run behind
    plotFreezeFrame(frames, 265, info, theme, features) # Frame 3: Assist 
    plotFreezeFrame(frames, 325, info, theme, features) # Frame 4: Shot

    # frame = tracks_smooth.iloc[45120]
    # # Display some vitality data from raw and smoothed data.
    # vit_data = getVitalityData(info, frames)
    # vitalityViz(tracks_raw[44750:45250], raw_vit_data, info, dark_theme)
    # vitalityViz(tracks_smooth[44750:45250], vit_data, info, dark_theme)
    
    # # HAM 4-0 IFE
    # frames = tracks_raw[57100:57600]
    # raw_vit_data = getVitalityData(info, frames)
    # balls = makingMovies("HAM-IFE 4-0 NOFILT.mp4", frames, light_theme, info, features)
    frames = tracks_smooth[57100:57600]
    
    if movies:
    
        print("Making movie of HAM-IFE 4-0...")
        balls = makingMovies("HAM-IFE 4-0.mp4", frames, theme, info, features)
        print("Done!")
        
    # vit_data = getVitalityData(info, frames)
    # vitalityViz(tracks_raw[57100:57600], raw_vit_data, info, dark_theme)
    # vitalityViz(tracks_smooth[57100:57600], vit_data, info, dark_theme)
    
    # HAM 5-0 IFE
    # print("Calculating velocity, speed and acceleration...")
    # tracks_smooth = calcVelocities(tracks2, hids, aids, filtering=True, interp_nans=True)
    # frames = tracks_smooth[25875:26250]
    
    # if movies:
        
    #     print("Making movie of HAM-IFE 5-0...")
    #     balls = makingMovies("HAM-IFE 5-0.mp4", frames, dark_theme, info, features)
    #     print("Done!")

    # print("Getting vitality data for frames...")
    # vit_data = getVitalityData(info, frames)
    # vitalityViz(frames, info, dark_theme)


def vizHAM_ORE(ht, at, movies=False):
 
    theme = DarkTheme(HAM_COL, ORE_COL)
    # theme = LightTheme(HAM_COL, ORE_COL)

    theme.padRight = -77
    features = FreezeFrameFeatures(True)
    features.voronoi = False
    features.pitchControl = False
    features.playerTracks = 5
    
    # Load first match in list.
    print("Loading first half data for HAM-ORE...")
    events1, info, stats1, tracks1 = getSignalityData(ht, at, 1)
    # events2, info2, stats2, tracks2 = getSignalityData(ht, at, 2)
    
    # Get team lineups.
    hids = info.homeTeamLineup.jersey_number.values
    aids = info.awayTeamLineup.jersey_number.values

    # Interpolate ball positions. This is bad, but it is what it is.
    tracks1.ball_pos_x.interpolate(method='linear', inplace=True)
    tracks1.ball_pos_y.interpolate(method='linear', inplace=True)
    # tracks2.ball_pos_x.interpolate(method='linear', inplace=True)
    # tracks2.ball_pos_y.interpolate(method='linear', inplace=True)
    
    # Calculate velocities without filterting.
    # tracks_raw = calcVelocities(tracks1, hids, aids, filtering=False, interp_nans=True)
    tracks_smooth = calcVelocities(tracks1, hids, aids, filtering=True, interp_nans=True)
        
    # HAM 1-1 ORE
    # frames = tracks_raw[57750:58500]
    
    # if movies:
        
    #     balls = makingMovies("HAM-ORE 1-1 NOFILT.mp4", frames, dark_theme, info, features)
    
    frames = tracks_smooth[57750:58500]
    errors = [(2334014, 2338335)]
    
    if movies:
        
        balls = makingMovies("HAM-ORE 1-1.mp4", frames, theme, info, features)
    
    plotFreezeFrame(frames, 565, info, theme, features)
    
    vitalityViz(frames, info, theme, 565, errors)
    
    # SECOND HALF DATA CORRUPTED!
    # tracks_smooth = calcVelocities(tracks2, hids, aids, filtering=True, interp_nans=True)
    # HAM 2-1 ORE
    # frames = tracks_smooth[18500:19000]
    # balls = makingMovies("HAM-ORE 2-1.mp4", frames, dark_theme, info, features)


def vizHAM_MFF(ht, at, movies=False):
 
    theme = DarkTheme(HAM_COL, MFF_COL)
    # theme = LightTheme(HAM_COL, MFF_COL)
    theme.padRight = -77
    features = FreezeFrameFeatures(True)
    features.voronoi = False
    features.pitchControl = False
    features.playerTracks = 5
    
    # Load first match in list.
    print("Loading first half data HAM-MFF...")
    events1, info, stats1, tracks1 = getSignalityData(ht, at, 1)
    # print("Loading second half data HAM-MFF...")
    # events2, info2, stats2, tracks2 = getSignalityData(ht, at, 2)
    
    # Get team lineups.
    hids = info.homeTeamLineup.jersey_number.values
    aids = info.awayTeamLineup.jersey_number.values

    # Interpolate ball positions. This is bad, but it is what it is.
    print("Interpolating ball positions...")
    tracks1.ball_pos_x.interpolate(method='linear', inplace=True)
    tracks1.ball_pos_y.interpolate(method='linear', inplace=True)
    # tracks2.ball_pos_x.interpolate(method='linear', inplace=True)
    # tracks2.ball_pos_y.interpolate(method='linear', inplace=True)
    
    # Calculate velocities without filterting.
    print("Calculating velocity, speed and accelerations...")
    # tracks_raw = calcVelocities(tracks1, hids, aids, filtering=False, interp_nans=True)
    tracks_smooth = calcVelocities(tracks1, hids, aids, filtering=True, interp_nans=True)
        
    # HAM 1-0 MFF
    # frames = tracks_raw[20750:21125]
    
    # if movies:
        
    #     features.voronoi = True
    #     balls = makingMovies("HAM-MFF 1-0 NOFILT.mp4", frames, dark_theme, info, features)
    #     features.voronoi = False
        
    frames = tracks_smooth[20750:21125]
    
    if movies:
        
        features.voronoi = True
        balls = makingMovies("HAM-MFF 1-0.mp4", frames, theme, info, features)
        features.voronoi = False

    vit_data = getVitalityData(info, frames)
    vitalityViz(frames, info, theme, 200)
    plotFreezeFrame(frames, 200, info, theme, features)

    vit_data = getVitalityData(info, tracks_smooth)
    vitalityVizSummary(vit_data, info, theme)

    # tracks_smooth = calcVelocities(tracks2, hids, aids, filtering=True, interp_nans=True)
    # HAM 2-0 MFF
    # frames = tracks_smooth[64000:64500]
    
    # if movies:
        
        # balls = makingMovies("HAM-MFF 2-0.mp4", frames, dark_theme, info, features)



###############################################################################
### -------------------------- End of Viz Methods ------------------------- ###
###############################################################################

if __name__ == "__main__":
    
    """
    HAM 5 - 2 IFE
    HAM: 18" (7), 28" (OG), 31" (7), 39" (22), 63" (20)
    IFE: 45+1", 47"
    
    HAM 5 - 1 ORE
    HAM: 39" (11), 57" (40), 62" (11), 80" (11), 90+3" (77)
    ORE: 11" (9)
    
    HAM 2 - 0 MFF
    HAM: 15" (20), 88" (4)
        2-0 is a set piece, tracking data useless for ball.
    """
     
    # List of matches for convenience.
    matches = [(TeamMap.HAM, TeamMap.IFE), # Goals at 38:24, 62:29 - 27:27, 45:20, 46:40, 
               (TeamMap.HAM, TeamMap.ORE),
               (TeamMap.HAM, TeamMap.MFF)]
    
    ht, at = matches[0]
    vizHAM_IFE(ht, at, movies=False)
    
    ht, at = matches[1]
    vizHAM_ORE(ht, at, movies=False)
    
    ht, at = matches[2]
    vizHAM_MFF(ht, at, movies=False)
    