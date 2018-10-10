#for dota api: https://gist.github.com/essramos/dbac40593b64e2193f2be68232f86b58
#adjust for use on league games

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 21:12:56 2018

@author: M29480
"""
# data source: http://oracleselixir.com/match-data/

import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

Probability(1847, 1723)
 
# Function to calculate win probability based on Elo rating
def Probability(rating1, rating2):
 
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))

np.log(abs(4) + 1)

# Function to update Elo rating based on match results 
def EloRating(Ra, Rb, K, d, kill, death):
  
    Pb = Probability(Ra, Rb)
    Pa = Probability(Rb, Ra)
    mov = kill - death
    adj = np.log(abs(mov) + 1)
    
    if (d == 1) :
        Ra = Ra + adj * K * (1 - Pa)
        Rb = Rb + adj * K * (0 - Pb)
     
    else :
        Ra = Ra + adj * K * (0 - Pa)
        Rb = Rb + adj * K * (1 - Pb)
     
    return Ra, Rb    

# Function to calculate the win probability in a best of five series based on Elo ratings
def bestofFive(team1, team2):
    elo1 = wcTeam['elo'][(wcTeam['team'] == team1)].values[0]
    elo2 = wcTeam['elo'][(wcTeam['team'] == team2)].values[0]
    prob = Probability(elo2, elo1)
    prob = prob**3 + 3*prob**3*(1-prob) + 6*prob**3*(1-prob)**2
    return prob

# Function to calculate win probability in the middle of best of five competition
def bestofFiveTourney(team1, team2, gamesPlayed, team1wins):
    elo1 = wcTeam['elo'][(wcTeam['team'] == team1)].values[0]
    elo2 = wcTeam['elo'][(wcTeam['team'] == team2)].values[0]
    prob = Probability(elo2, elo1)
    if gamesPlayed == 0:
        prob = prob**3 + 3*prob**3*(1-prob) + 6*prob**3*(1-prob)**2
    elif (gamesPlayed == 1) & (team1wins == 0):
        prob = prob**3 + 3*prob**3*(1-prob)
    elif (gamesPlayed == 2) & (team1wins == 0):
        prob = prob**3
    elif (gamesPlayed == 1) & (team1wins == 1):
        prob = prob**2 + 2*prob**2*(1-prob) + 2*prob**2*(1-prob**2)
    elif (gamesPlayed == 2) & (team1wins == 1):
        prob = prob**2 + 2*prob**2*(1-prob)
    elif (gamesPlayed == 3) & (team1wins == 1):
        prob = prob**2
    elif (gamesPlayed == 3) & (team1wins == 2):
        prob = prob + prob*(1-prob)
    elif (gamesPlayed == 4) & (team1wins == 2):
        prob = prob
    return prob

# Load league files
summer18 = pd.read_csv('2018Summer.csv')
summer18['year'] = 2018
spring18 = pd.read_csv('2018Spring.csv')
spring18['year'] = 2018
complete17 = pd.read_csv('2017Complete.csv')
complete17['year']=2017
complete16 = pd.read_csv('2016Complete.csv')
complete16['year']= 2016
teamid = pd.read_csv('teamIDs.csv')
teamid['teamid'] = teamid['teamid'].astype('str')
leagueList = pd.read_csv('leagueList.csv')
playinResults = pd.read_csv('wcPlayinResults.csv')
worldResults = pd.read_csv('wcResults.csv')

worldResults = pd.concat([worldResults, playinResults])

# Concatenate league files
full_lol = pd.concat([pd.concat([pd.concat([complete16, complete17]), spring18]), summer18])
teamids = teamid[['teamid', 'team']]
full_lol = full_lol.merge(teamids, on='team')
full_lol['teamid'] = full_lol['teamid'].astype('str')

full_lol = full_lol[['split', 'year', 'league', 'gameid', 'date', 'week', 'game', 'playerid', 'teamid', 'team', 'result', 'k', 'd']]

# Sekect only team part of results -> no current analysis on individual players
full_lol = full_lol[(full_lol['playerid'] == 100) | (full_lol['playerid'] == 200)]

# Fix dates and sort games by date (so Elo calculates sequentially)
full_lol['date'] = pd.to_datetime(full_lol['date'])
full_lol = full_lol.sort_values(by=['date'])

# Build dataframe of teams
teams = full_lol['teamid'].unique()
teams = np.append(teams, '204')
teams = np.append(teams, '205')
teamELO = pd.DataFrame(teams)
teamELO = teamELO.rename(index=str, columns = {0: 'teamid'})
teamid = teamid[['teamid']]
teamid = teamid['teamid'].unique()
teamid=pd.DataFrame(teamid)
teamid = teamid.rename(index=str, columns = {0: 'teamid'})
teamELO = teamid.merge(teamELO, on='teamid')
leagueList['teamid'] = leagueList['teamid'].astype('str')
teamELO = teamELO.merge(leagueList, on='teamid')

# Set default Elo for all teams -> league dependent due to talent discrepancies
teamELO['elo'] = 1475 #1600 is an abitrary number and based on other popular ELO systems
teamELO['elo'][(teamELO['league'] == 'LCK')] = 1800
teamELO['elo'][(teamELO['league'] == 'LPL')] = 1800

#leagues = full_lol['league'].unique()
#leagues=pd.DataFrame(leagues) 

# Coombine all games onto one line by making one team the opponent
full_lol['opp'] = ""
for game in full_lol['gameid'].unique():
    team2 = full_lol['teamid'][(full_lol['playerid'] == 200) & (full_lol['gameid'] == game)]
    full_lol['opp'][(full_lol['playerid'] == 100) & (full_lol['gameid'] == game)] = team2.values[0]

##### This would be a good place to automate model to speed up
    
# Down select to one row per game
full_lol = full_lol[full_lol['playerid'] == 100]

# Add results from world championship into game list -> compiled me so less data per game
worldResults = worldResults.merge(teamids, on = 'team')
oppid = teamids
oppid = oppid.rename(index=str, columns = {'team': 'opp', 'teamid': 'oppid'})
worldResults = worldResults.merge(oppid, on='opp')
worldResults = worldResults.drop(['opp', 'group'], axis=1)
worldResults = worldResults.rename(index=str, columns = {'oppid': 'opp'})
worldResults = worldResults[(worldResults['played'] == 'X')]
worldResults = worldResults.drop(['played'], axis=1)
full_lol = full_lol.drop(['playerid'], axis=1)
worldResults['teamid'] = worldResults['teamid'].astype('str')

full_lol = pd.concat([full_lol, worldResults])


full_lol['date'] = pd.to_datetime(full_lol['date'])
full_lol = full_lol.sort_values(by=['date'])

full_lol['gameids'] = range(0, len(full_lol))


# Cycle through all games and update elo rankings accordingly
k = 15 #k needs to be tuned
for game in full_lol['gameids']:
    team1 = teamELO['teamid'] == full_lol['teamid'][(full_lol['gameids'] == game)].values[0]
    team2 = teamELO['teamid'] == full_lol['opp'][(full_lol['gameids'] == game)].values[0]
    elo1 = teamELO['elo'][team1].values[0]
    elo2 = teamELO['elo'][team2].values[0]
    result = full_lol['result'][(full_lol['gameids'] == game)].values[0]
    kill = full_lol['k'][(full_lol['gameids'] == game)].values[0]
    death = full_lol['d'][(full_lol['gameids'] == game)].values[0]

    newelo1, newelo2 = EloRating(elo1, elo2, k, result, kill, death)
    
    teamELO['elo'][team1] = newelo1
    teamELO['elo'][team2] = newelo2

# Build standings dataframe (for validation)
games = full_lol['teamid'].value_counts()
wins = full_lol.groupby('teamid')['result'].sum()
record = pd.concat([games, wins], axis=1)
record['losses'] = record['teamid'] - record['result']
record = record[['result', 'losses']]
record = record.rename(index=str, columns = {'result': 'Wins', 'losses': 'Losses'})
teamELO = teamELO.set_index('teamid')
teamELO = pd.concat([teamELO, record], axis=1)
teamELO = teamELO.reset_index()
teamELO = teamELO.rename(index=str, columns={'index': 'teamid'})
teamELO = teamELO.merge(teamid, on='teamid')
teamELO = teamELO.merge(teamids, on='teamid')
teamELO = teamELO[['team', 'teamid', 'elo', 'Wins', 'Losses', 'league']] #, 'numberOfLeagues', 'lastLeagues', 'lastYearPlayed']]
teamELO.to_csv('teamelo.csv')

# Analyze the World Cup Teams for LoL 2018 Worlds analysis
wcTeam = ['Cloud9', 'G2 Esports', 'EDward Gaming', 'MAD Team', 'Flash Wolves', '100 Thieves', 'Team Liquid', 
          'Vitality', 'Fnatic', 'Invictus Gaming', 'Royal Never Give Up', 'Gen.G', 'Afreeca Freecs', 
          'KT Rolster', 'Phong Vu Buffalo', 'G-Rex']

wcTeam = pd.DataFrame(wcTeam)
wcTeam = wcTeam.rename(index=str, columns={0: 'team'})
wcTeam = wcTeam.merge(teamELO, on='team')

# All analysis on first round must be done by group
wcGroup = pd.read_csv('wcGroups.csv')
wcTeam = wcTeam.merge(wcGroup, on='team')
wcTeams = wcTeam[(wcTeam.group == 'A') | (wcTeam.group == 'B') | (wcTeam.group == 'C') | (wcTeam.group == 'D')]
wcTeams['condWins'] = ''
wcTeams['currWins'] =''
wcTeams[['condWins', 'currWins']] = wcTeams[['condWins', 'currWins']].apply(pd.to_numeric)

# Read back in current results from games and predict results for all games based on Elo (uses current Elo, not at time of game)
worldResults = pd.read_csv('wcResults.csv')
worldResults['prob'] = ''
worldResults['pred'] = ''
worldResults['oppwin'] = 1 - worldResults['result']
for game in worldResults.gameid:
    worldResults['pred'][(worldResults.gameid == game)] = Probability(wcTeams['elo'][(wcTeams['team'] == worldResults['opp'][(worldResults.gameid == game)].values[0])].values[0], 
                wcTeams['elo'][(wcTeams['team'] == worldResults['team'][(worldResults.gameid == game)].values[0])])
    if worldResults.played[(worldResults.gameid == game)].values[0] == 'X':
        worldResults['prob'][(worldResults.gameid == game)] = worldResults['result'][(worldResults.gameid == game)]
    else:
        worldResults['prob'][(worldResults.gameid == game)] = ''
        worldResults['prob'][(worldResults.gameid == game)] = Probability(wcTeams['elo'][(wcTeams['team'] == worldResults['opp'][(worldResults.gameid == game)].values[0])].values[0], 
                    wcTeams['elo'][(wcTeams['team'] == worldResults['team'][(worldResults.gameid == game)].values[0])])

# Calculate the current and predicted wins for each group
for group in wcTeams.group.unique():
    #towin = towin['towin'][(towin['group'] == group)]
    team1 = wcTeams['team'][(wcTeams['teamNum'] == 1) & (wcTeams['group'] == group)].values[0]
    team2 = wcTeams['team'][(wcTeams['teamNum'] == 2) & (wcTeams['group'] == group)].values[0]
    team3 = wcTeams['team'][(wcTeams['teamNum'] == 3) & (wcTeams['group'] == group)].values[0]
    team4 = wcTeams['team'][(wcTeams['teamNum'] == 4) & (wcTeams['group'] == group)].values[0]
    game1 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team1) & (worldResults.opp == team2)].values[0]
    game2 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team3) & (worldResults.opp == team4)].values[0]
    game3 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team1) & (worldResults.opp == team4)].values[0]
    game4 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team2) & (worldResults.opp == team3)].values[0]
    game5 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team1) & (worldResults.opp == team3)].values[0]
    game6 = worldResults['prob'][(worldResults.week == 1) & (worldResults.team == team2) & (worldResults.opp == team4)].values[0]
    game7 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team1) & (worldResults.opp == team2)].values[0]
    game8 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team3) & (worldResults.opp == team4)].values[0]
    game9 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team1) & (worldResults.opp == team4)].values[0]
    game10 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team2) & (worldResults.opp == team3)].values[0]
    game11 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team1) & (worldResults.opp == team3)].values[0]
    game12 = worldResults['prob'][(worldResults.week == 2) & (worldResults.team == team2) & (worldResults.opp == team4)].values[0]
    prob1 = game1 + game3 + game5 + game7 + game9 + game11
    prob2 = (1- game1) + game4 + game6 + (1 - game7) + game10 + game12
    prob3 = game2 + (1 - game4) + (1 - game5) + game8 + (1 - game10)+ (1 - game11)
    prob4 = (1-game2) + (1-game3) + (1-game6) + (1-game8) + (1-game9) + (1-game12)
    wcTeams['condWins'][(wcTeams['team'] == team4)] = prob4
    wcTeams['condWins'][(wcTeams['team'] == team3)] = prob3
    wcTeams['condWins'][(wcTeams['team'] == team2)] = prob2
    wcTeams['condWins'][(wcTeams['team'] == team1)] = prob1
    
    if worldResults['played'][(worldResults.team == team1) | (worldResults.opp == team1)].str.count('X').sum() > 0:
        wcTeams['currWins'][(wcTeams['team'] == team1)] = worldResults['result'][(worldResults.team == team1)].sum() \
        + worldResults['oppwin'][(worldResults.opp == team1)].sum()
    else:
        wcTeams['currWins'][(wcTeams['team'] == team1)] = 0
    
    if worldResults['played'][(worldResults.team == team2) | (worldResults.opp == team2)].str.count('X').sum() > 0:
        wcTeams['currWins'][(wcTeams['team'] == team2)] = worldResults['result'][(worldResults.team == team2)].sum() \
        + worldResults['oppwin'][(worldResults.opp == team2)].sum()
    else:
        wcTeams['currWins'][(wcTeams['team'] == team2)] = 0
    
    if worldResults['played'][(worldResults.team == team3) | (worldResults.opp == team3)].str.count('X').sum() > 0:
        wcTeams['currWins'][(wcTeams['team'] == team3)] = worldResults['result'][(worldResults.team == team3)].sum() \
        + worldResults['oppwin'][(worldResults.opp == team3)].sum()
    else:
        wcTeams['currWins'][(wcTeams['team'] == team3)] = 0
    
    if worldResults['played'][(worldResults.team == team4) | (worldResults.opp == team4)].str.count('X').sum() > 0:
        wcTeams['currWins'][(wcTeams['team'] == team4)] = worldResults['result'][(worldResults.team == team4)].sum() \
        + worldResults['oppwin'][(worldResults.opp == team4)].sum()
    else:
        wcTeams['currWins'][(wcTeams['team'] == team4)] = 0
             
# Analyze the knockout rounds separately due to random draw
seeds = pd.read_csv('knockoutSeeds.csv')
seeds['sfodds'] = ''
seeds['fodds'] = ''
seeds['codds'] = ''

# Predict odds of making semifinals
for team in seeds.team:
    if seeds.seed[(seeds.team == team)].values[0] == 'A1':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'B2':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'B1':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0])    
    elif seeds.seed[(seeds.team == team)].values[0] == 'A2':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'C1':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'D2':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'D1':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0])
    elif seeds.seed[(seeds.team == team)].values[0] == 'C2':
        seeds.sfodds[(seeds.team == team)] = bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0])

# Predicts odds of making finals
for team in seeds.team:
    if seeds.seed[(seeds.team == team)].values[0] == 'A1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.sfodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.sfodds[(seeds.seed == 'C2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]
    elif seeds.seed[(seeds.team == team)].values[0] == 'B2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.sfodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.sfodds[(seeds.seed == 'C2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]
    elif seeds.seed[(seeds.team == team)].values[0] == 'D1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.sfodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.sfodds[(seeds.seed == 'B2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'C2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.sfodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.sfodds[(seeds.seed == 'B2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'B1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.sfodds[(seeds.seed == 'D2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.sfodds[(seeds.seed == 'C1')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'A2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.sfodds[(seeds.seed == 'D2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.sfodds[(seeds.seed == 'C1')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'C1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.sfodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.sfodds[(seeds.seed == 'A2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'D2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.sfodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.sfodds[(seeds.seed == 'A2')].values[0]
        seeds.fodds[(seeds.team == team)]  = winprob * seeds.sfodds[(seeds.team == team)]

# Predict odds of winning title
for team in seeds.team:
    if seeds.seed[(seeds.team == team)].values[0] == 'A1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.fodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.fodds[(seeds.seed == 'A2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.fodds[(seeds.seed == 'C1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.fodds[(seeds.seed == 'D2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]
    elif seeds.seed[(seeds.team == team)].values[0] == 'B2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.fodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.fodds[(seeds.seed == 'A2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.fodds[(seeds.seed == 'C1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.fodds[(seeds.seed == 'D2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]
    elif seeds.seed[(seeds.team == team)].values[0] == 'D1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.fodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.fodds[(seeds.seed == 'A2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.fodds[(seeds.seed == 'C1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.fodds[(seeds.seed == 'D2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'C2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'B1')].values[0]) * seeds.fodds[(seeds.seed == 'B1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'A2')].values[0]) * seeds.fodds[(seeds.seed == 'A2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C1')].values[0]) * seeds.fodds[(seeds.seed == 'C1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D2')].values[0]) * seeds.fodds[(seeds.seed == 'D2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]     
    elif seeds.seed[(seeds.team == team)].values[0] == 'B1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.fodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.fodds[(seeds.seed == 'B2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.fodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.fodds[(seeds.seed == 'C2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]      
    elif seeds.seed[(seeds.team == team)].values[0] == 'A2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.fodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.fodds[(seeds.seed == 'B2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.fodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.fodds[(seeds.seed == 'C2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]        
    elif seeds.seed[(seeds.team == team)].values[0] == 'C1':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.fodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.fodds[(seeds.seed == 'B2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.fodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.fodds[(seeds.seed == 'C2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]       
    elif seeds.seed[(seeds.team == team)].values[0] == 'D2':
        winprob = bestofFive(team, seeds.team[(seeds.seed == 'A1')].values[0]) * seeds.fodds[(seeds.seed == 'A1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'B2')].values[0]) * seeds.fodds[(seeds.seed == 'B2')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'D1')].values[0]) * seeds.fodds[(seeds.seed == 'D1')].values[0] \
        + bestofFive(team, seeds.team[(seeds.seed == 'C2')].values[0]) * seeds.fodds[(seeds.seed == 'C2')].values[0]
        seeds.codds[(seeds.team == team)]  = winprob * seeds.fodds[(seeds.team == team)]      
        
        
###################### Visualizations

# Next Day's Games    
tomorrowGames = worldResults[(worldResults.date == '10/11/2018')]
c=0
x = len(tomorrowGames)
f, axes = plt.subplots(x,1, sharex='none', sharey='row')
f.text(0.58, 1, 'Win Probability', ha='center')
for n in tomorrowGames.index:
    game = tomorrowGames.loc[n]
    team1 = game.team
    team2 = game.opp
    elo1 = teamELO['elo'][(teamELO['team'] == team1)].values[0]
    elo2 = teamELO['elo'][(teamELO['team'] == team2)].values[0]
    prob1 = Probability(elo2, elo1) * 100
    prob2 = Probability(elo1, elo2) * 100
    gameData = {'Team': [team1, team2],'Win Probability': [prob1, prob2]}
    gameData = pd.DataFrame(data=gameData)
    gameData = gameData[['Team', 'Win Probability']]
    gameData = gameData.set_index('Team')
    gameData.index.names = ['']
    gameData = gameData.rename(columns={'Win Probability': ''})
    ax = sns.heatmap(gameData, annot=True, vmin=0, vmax=100, fmt=".2f", ax=axes[c], cbar=False, cmap = "GnBu")
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    c = c + 1
plt.savefig('tomorrowGames.png', bbox_inches = "tight")
plt.clf()
    
# Current Elo Rankings
wcElo = wcTeam[['team', 'elo', 'league', 'group']]


wcElo = wcElo.rename(columns={'team': 'Team', 'elo': 'Elo', 'league': 'League', 'group': 'Group'})

wcElo = wcElo[['Team', 'Elo']]
wcElo= wcElo.set_index(['Team'])
wcElo = wcElo.sort_values(by=['Elo'], ascending=False)
ax = sns.heatmap(wcElo, annot=True, fmt=".2f", cmap = "BuGn")
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.yaxis.label.set_visible(False)
plt.savefig('currentElo.png', bbox_inches = "tight")
plt.clf()

# Conditional Wins per Group
groups = wcTeams[['team', 'group', 'condWins', 'currWins']]
groups = groups.sort_values(by=['condWins'], ascending=False)
groups = groups.rename(columns={'group': 'Group', 'team': 'Team', 'currWins': 'Current Wins', 'condWins': 'Predicted Wins'})
c = 0
f,axes = plt.subplots(4,1, sharex='col', sharey='row')
f.text(0.41, 1, 'Current Wins', ha='center')
f.text(0.78, 1, 'Predicted Wins', ha='center')
groups = groups.sort_values(by=['Group'], ascending=True) 
f.subplots_adjust(wspace=0.01)
for group in groups.Group.unique():
    teams = groups[(groups['Group'] == group)]
    teams = teams[['Team', 'Current Wins', 'Predicted Wins']]
    teams = teams.rename(columns={'Team': group})
    teams = teams.set_index(group)
    teams['Current Wins'] = pd.to_numeric(teams['Current Wins'])
    teams['Predicted Wins'] = pd.to_numeric(teams['Predicted Wins'])
    teams = teams.rename(columns={'Current Wins': '', 'Predicted Wins': ''})
    ax = sns.heatmap(teams, annot=True, ax=axes[c], fmt=".2f", cbar=False, cmap = "YlGnBu")
    ax.yaxis.label.set_visible(False)
    c = c + 1
plt.savefig('groupStand.png', bbox_inches = "tight")
plt.clf()

# Title probability in knockout round
wcProb = seeds[['team','sfodds', 'fodds', 'codds']]
wcProb.sfodds = wcProb.sfodds.astype('float') * 100
wcProb.fodds = wcProb.fodds.astype('float') * 100
wcProb.codds = wcProb.codds.astype('float') * 100
wcProb = wcProb.rename(columns={'team': 'Team', 'sfodds': 'Semifinal Probability', 'fodds': 'Final Probability', 'codds': 'Title Probability'})

wcProb = wcProb[['Team', 'Semifinal Probability', 'Final Probability', 'Title Probability']]
wcProb= wcProb.set_index(['Team'])
wcProb = wcProb.sort_values(by=['Title Probability'], ascending=False)
ax = sns.heatmap(wcProb, annot=True, vmin=0, vmax=100,fmt=".2f", cmap = "Blues")
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
ax.yaxis.label.set_visible(False)
plt.savefig('titleOdds.png', bbox_inches = "tight")
plt.clf()

'''
# would like to list group and league with Elo -> needs a subplot
wcElo = wcElo[['Team', 'Group', 'League', 'Elo']]
wcElo= wcElo.set_index(['Team', 'Group', 'League'])
wcElo = wcElo.sort_values(by=['Elo'], ascending=False)
sns.heatmap(wcElo, annot=True, fmt=".2f")
'''
